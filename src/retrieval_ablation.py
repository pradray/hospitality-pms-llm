"""Retrieval ablations: measure what the retriever can actually surface.

Motivation (from the Phase 2 test-set analysis): the generator is no longer the
main bottleneck for API orchestration — only 50% of orchestration tasks had *all*
their gold API paths inside the top-5, and 27% of gold paths were never retrieved
at all. No amount of fine-tuning fixes an endpoint the model never sees.

This scores retrieval *directly* — gold-path recall and gold content-word coverage
— with no LLM in the loop, so a whole sweep runs in seconds instead of hours.
Generation is only worth running for whichever configuration wins here.

IMPORTANT: sweeps run on the DEV split. The 46-task test split stays untouched;
tuning retrieval on it would void the held-out claim that the Phase 2 results rest
on. Run the winner against test exactly once, at the end.

Variants:
  dense       - current system: bge-small over ChromaDB, cosine top-k
  bm25        - lexical only; strong on rare exact identifiers (getProfileByExtId)
                that dense embeddings tend to blur
  hybrid      - Reciprocal Rank Fusion of dense + BM25
  rerank      - dense/hybrid recall@N, reordered by a cross-encoder
  multiquery  - split the question into sub-queries and union the results, for
                multi-step workflow questions where one embedding cannot possibly
                retrieve four different endpoints

Usage:
    python src/retrieval_ablation.py --split dev
    python src/retrieval_ablation.py --split dev --variants dense,hybrid --topk 5,10
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag_pipeline import EMBEDDING_MODEL, COLLECTION_NAME, VECTORSTORE_DIR
from analyze_results import extract_paths, normalise_path, VERSIONED_PATH_RE

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHUNK_FILES = [
    OUTPUT_DIR / "api_chunks" / "all_endpoints.jsonl",
    OUTPUT_DIR / "doc_chunks" / "all_doc_chunks.jsonl",
    OUTPUT_DIR / "postman_chunks" / "all_postman_chunks.jsonl",
]

RERANKER_MODEL = "BAAI/bge-reranker-base"
RRF_K = 60  # standard Reciprocal Rank Fusion constant


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, plus camelCase splitting.

    Operation ids like `getProfileByExtId` are exactly the tokens dense retrieval
    handles worst, so BM25 gets both the whole identifier and its parts.
    """
    text = text or ""
    parts = re.findall(r"[A-Za-z0-9_]+", text)
    out = []
    for p in parts:
        out.append(p.lower())
        for piece in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", p):
            if len(piece) > 1:
                out.append(piece.lower())
    return out


class Corpus:
    def __init__(self):
        self.texts, self.ids = [], []
        for fn in CHUNK_FILES:
            if not fn.exists():
                continue
            with open(fn) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("text"):
                        self.texts.append(d["text"])
                        self.ids.append(d.get("id", str(len(self.ids))))
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        return self._bm25


class Retrievers:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    def dense(self, query: str, k: int) -> list[str]:
        emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=[emb], n_results=k,
                                    include=["documents"])
        return res["documents"][0]

    def bm25(self, query: str, k: int) -> list[str]:
        scores = self.corpus.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.corpus.texts[i] for i in top]

    def hybrid(self, query: str, k: int, pool: int = 30) -> list[str]:
        """Reciprocal Rank Fusion — combines rankings without needing the two
        score scales (cosine distance vs BM25) to be comparable."""
        d = self.dense(query, pool)
        b = self.bm25(query, pool)
        scores = defaultdict(float)
        for rank, doc in enumerate(d):
            scores[doc] += 1.0 / (RRF_K + rank + 1)
        for rank, doc in enumerate(b):
            scores[doc] += 1.0 / (RRF_K + rank + 1)
        return [doc for doc, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]

    def rerank(self, query: str, k: int, pool: int = 30, base: str = "dense") -> list[str]:
        cands = self.hybrid(query, pool) if base == "hybrid" else self.dense(query, pool)
        if not cands:
            return []
        scores = self.reranker.predict([(query, c) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        return [cands[i] for i in order[:k]]

    def multiquery(self, query: str, k: int, base: str = "dense") -> list[str]:
        """Split a multi-step question into sub-queries and union the results.

        Workflow questions ask for a *sequence* of endpoints; a single embedding
        is pulled toward one of them, so the others are never retrieved.
        Rule-based splitting (no LLM) keeps this cheap and reproducible.
        """
        subs = [query]
        parts = re.split(r"(?:\band then\b|\bthen\b|\bafter that\b|[;\n]|\d\.\s)", query)
        subs += [p.strip() for p in parts if len(p.strip()) > 25]
        seen, out = set(), []
        per = max(2, k // max(1, min(len(subs), 4)))
        for s in subs[:4]:
            hits = self.hybrid(s, per) if base == "hybrid" else self.dense(s, per)
            for h in hits:
                if h not in seen:
                    seen.add(h)
                    out.append(h)
        return out[:k]


def gold_targets(task: dict):
    gold_paths = {normalise_path(p) for p in extract_paths(task["expected_answer"])}
    gold_paths = {g for g in gold_paths if "/v" in g}
    words = {w for w in re.findall(r"[a-z]{5,}", task["expected_answer"].lower())}
    return gold_paths, words


def score_context(chunks: list[str], gold_paths: set, gold_words: set):
    ctx = "\n".join(chunks)
    ctx_paths = {normalise_path(p) for p in VERSIONED_PATH_RE.findall(ctx)}
    ctx_low = ctx.lower()
    found = len(gold_paths & ctx_paths)
    complete = 1 if gold_paths and gold_paths <= ctx_paths else 0
    cov = (sum(1 for w in gold_words if w in ctx_low) / len(gold_words)) if gold_words else 0.0
    return found, len(gold_paths), complete, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="Sweep on dev. Only touch test for the final chosen config.")
    ap.add_argument("--variants", default="dense,bm25,hybrid,rerank,multiquery")
    ap.add_argument("--topk", default="5,10,15")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    if args.split == "test":
        print("!! Running on the HELD-OUT TEST split. Only do this once, for the\n"
              "!! configuration already chosen on dev.\n")

    tasks = [json.loads(l) for l in
             open(DATA_DIR / f"benchmark_{args.split}.jsonl")]
    print(f"Loaded {len(tasks)} {args.split} tasks")

    corpus = Corpus()
    print(f"Corpus: {len(corpus.texts)} chunks")
    R = Retrievers(corpus)

    variants = [v.strip() for v in args.variants.split(",")]
    topks = [int(k) for k in args.topk.split(",")]

    results = {}
    print(f"\n{'Variant':<14}{'k':>4}{'path recall':>14}{'complete':>11}"
          f"{'content cov':>13}{'orch recall':>13}")
    print("-" * 70)

    for variant in variants:
        for k in topks:
            tot_found = tot_gold = tot_complete = tot_withpaths = 0
            covs = []
            orch_found = orch_gold = 0
            for t in tasks:
                gp, gw = gold_targets(t)
                if variant == "dense":
                    chunks = R.dense(t["question"], k)
                elif variant == "bm25":
                    chunks = R.bm25(t["question"], k)
                elif variant == "hybrid":
                    chunks = R.hybrid(t["question"], k)
                elif variant == "rerank":
                    chunks = R.rerank(t["question"], k)
                elif variant == "multiquery":
                    chunks = R.multiquery(t["question"], k)
                else:
                    raise SystemExit(f"unknown variant {variant}")

                found, ngold, complete, cov = score_context(chunks, gp, gw)
                covs.append(cov)
                if ngold:
                    tot_found += found
                    tot_gold += ngold
                    tot_complete += complete
                    tot_withpaths += 1
                    if t.get("category") == "api_orchestration":
                        orch_found += found
                        orch_gold += ngold

            recall = 100 * tot_found / tot_gold if tot_gold else 0
            comp = 100 * tot_complete / tot_withpaths if tot_withpaths else 0
            cov = 100 * sum(covs) / len(covs)
            orec = 100 * orch_found / orch_gold if orch_gold else 0
            key = f"{variant}@{k}"
            results[key] = {"path_recall": recall, "complete_pct": comp,
                            "content_cov": cov, "orch_recall": orec,
                            "n_with_paths": tot_withpaths}
            print(f"{variant:<14}{k:>4}{recall:>13.0f}%{comp:>10.0f}%"
                  f"{cov:>12.0f}%{orec:>12.0f}%")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_out}")

    best = max(results, key=lambda k: results[k]["path_recall"])
    print(f"\nBest path recall: {best} ({results[best]['path_recall']:.0f}%)")


if __name__ == "__main__":
    main()
