"""Build the QLoRA supervised fine-tuning dataset (Phase 2).

Goal: *grounding*. The Phase-1 error analysis showed the retriever surfaces the
right chunk (win/loss retrieval distance is ~identical, 0.213 vs 0.215) but the
generator ignores it and hallucinates API paths (48% of all errors). This script
builds instruction pairs that teach the model to answer *faithfully to the
retrieved context*, in the exact prompt format used at inference time.

Three sources of supervision:

  1. API-path grounding pairs (programmatic, high volume).
     Questions are generated from API-endpoint / Postman chunks; gold answers are
     mechanically derived from chunk fields (method, path, operationId). This gives
     label-noise-free supervision on exactly the thing the model hallucinates.

  2. Config-advisory + troubleshooting pairs (human-curated).
     The dev-benchmark items for these two categories carry rich, multi-point gold
     answers that cannot be synthesised mechanically. Upweighted so reasoning-style
     answers are not drowned out by the templated API pairs.

  3. Dev-benchmark api_orchestration pairs (human-curated), for natural phrasing.

Every example is wrapped in the *same* format the RAG pipeline uses at inference:
system = SYSTEM_PROMPT_RAG, user = "Context:\n[Source i ...]\n{chunk}\n---...\n\n
Question: {q}", assistant = gold. Context is built from real top-5 retrieval over
the ChromaDB vector store, so the model trains on the same distractor-laden context
it will see when evaluated. The gold source chunk is injected if retrieval misses it,
so every example is answerable from its context.

The held-out 46-task TEST split is never touched here.

Run (local, no GPU needed):
    .venv/bin/python src/build_sft_dataset.py
Outputs:
    data/sft_train.jsonl   (chat-format: {"messages": [...]})
    data/sft_val.jsonl
    data/sft_stats.json
"""

import json
import random
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

from rag_pipeline import (
    SYSTEM_PROMPT_RAG,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    VECTORSTORE_DIR,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

API_CHUNKS = OUTPUT_DIR / "api_chunks" / "all_endpoints.jsonl"
POSTMAN_CHUNKS = OUTPUT_DIR / "postman_chunks" / "all_postman_chunks.jsonl"
DEV_BENCHMARK = DATA_DIR / "benchmark_dev.jsonl"

SEED = 42
TOP_K = 5
VAL_FRACTION = 0.05

# Training sequence budget. ~3.6 chars/token on this API-heavy text, so 14,700
# chars ≈ 4,096 tokens — what a 16 GB T4 can train at 4-bit with grad checkpointing.
MAX_SEQ_TOKENS = 4096
MAX_CHARS = int(MAX_SEQ_TOKENS * 3.6)

# How many programmatic API-path pairs to sample, and category upweighting for the
# human-curated dev pairs. Kept modest: LoRA on a 3B overfits on a large, templated
# set, and the point is grounding behaviour, not memorising every endpoint.
N_API_PAIRS = 650
N_POSTMAN_PAIRS = 150
UPWEIGHT = {
    "config_advisory": 3,
    "troubleshooting": 3,
    "api_orchestration": 2,
}

random.seed(SEED)


# --------------------------------------------------------------------------- #
# Retrieval (LLM-free): mirror RAGPipeline.retrieve / _build_user_prompt so the
# training prompts are byte-for-byte what the eval harness will feed the model,
# without loading the 2 GB GGUF.
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query, top_k=TOP_K):
        emb = self.embedder.encode(query, normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({"text": doc, "meta": meta, "dist": dist})
        return out


def build_user_prompt(query, chunks):
    """Identical formatting to RAGPipeline._build_user_prompt."""
    if not chunks:
        return query
    parts = []
    for i, ch in enumerate(chunks, 1):
        meta = ch["meta"]
        source = meta.get("source_file", "")
        section = meta.get("section", meta.get("path", ""))
        header = f"[Source {i}: {source} — {section}]"
        parts.append(f"{header}\n{ch['text']}")
    context = "\n\n---\n\n".join(parts)
    return f"Context:\n{context}\n\nQuestion: {query}"


def context_with_gold(retriever, query, gold_text):
    """Top-k retrieval; guarantee the gold source chunk is present so the example
    is answerable. If missing, inject it at a random slot and drop the last."""
    chunks = retriever.retrieve(query, TOP_K)
    if gold_text is not None:
        texts = [c["text"] for c in chunks]
        if gold_text not in texts:
            slot = random.randrange(TOP_K)
            chunks = chunks[: TOP_K - 1]
            chunks.insert(
                slot, {"text": gold_text, "meta": {}, "dist": 0.0, "protected": True}
            )
    return chunks


# --------------------------------------------------------------------------- #
# Source 1 — programmatic API-path grounding pairs
# --------------------------------------------------------------------------- #
API_Q_TEMPLATES = [
    "Which OHIP API endpoint would you call to {action}? Give the HTTP method and full path.",
    "In OPERA Cloud / OHIP, how do I {action} via the API? Specify the method, path, and operation.",
    "What is the operation ID and endpoint path used to {action}?",
]


def _action_phrase(summary, description):
    """Turn a chunk summary into a lower-cased action phrase for a question."""
    s = (summary or "").strip()
    if not s:
        s = (description or "").strip().split(".")[0]
    s = re.sub(r"\s+", " ", s)
    s = s[0].lower() + s[1:] if s else s
    return s.rstrip(".")


def _api_gold(method, path, op_id, description, required_params):
    lines = [f"`{method} {path}`" + (f" (operationId `{op_id}`)" if op_id else "") + "."]
    desc = (description or "").strip()
    desc = re.sub(r"\s+OperationId:.*$", "", desc, flags=re.I).strip()
    if desc:
        lines.append(desc if desc.endswith(".") else desc + ".")
    if required_params:
        lines.append("Required parameters: " + ", ".join(required_params) + ".")
    return " ".join(lines)


def load_api_pairs(retriever):
    rows = [json.loads(l) for l in open(API_CHUNKS)]
    # dedup by operation_id (keep one endpoint per operation), balance across modules
    by_module = {}
    seen_ops = set()
    for r in rows:
        op = r.get("operation_id") or r.get("path")
        if op in seen_ops:
            continue
        seen_ops.add(op)
        if not r.get("path") or not r.get("method"):
            continue
        by_module.setdefault(r.get("module", "multi"), []).append(r)

    modules = list(by_module.keys())
    for m in modules:
        random.shuffle(by_module[m])

    # round-robin across modules for balance
    picked = []
    idx = {m: 0 for m in modules}
    while len(picked) < N_API_PAIRS and any(idx[m] < len(by_module[m]) for m in modules):
        for m in modules:
            if idx[m] < len(by_module[m]):
                picked.append(by_module[m][idx[m]])
                idx[m] += 1
                if len(picked) >= N_API_PAIRS:
                    break

    pairs = []
    for r in picked:
        action = _action_phrase(r.get("summary"), r.get("description"))
        if not action or len(action) < 4:
            continue
        req_params = [
            p["name"]
            for p in r.get("parameters", [])
            if isinstance(p, dict) and p.get("required")
        ][:6]
        q = random.choice(API_Q_TEMPLATES).format(action=action)
        gold = _api_gold(
            r["method"], r["path"], r.get("operation_id"),
            r.get("description") or r.get("summary"), req_params,
        )
        # Inject this endpoint's own chunk if top-k retrieval missed it. Without
        # this, ~21% of pairs would ask the model to emit a path that is absent
        # from its context — training the exact hallucination we are trying to cure.
        chunks = context_with_gold(retriever, q, r.get("text"))
        pairs.append(("api_orchestration", r.get("module", "multi"), q, gold, chunks))
    return pairs


def load_postman_pairs(retriever):
    rows = [json.loads(l) for l in open(POSTMAN_CHUNKS)]
    random.shuffle(rows)
    pairs = []
    for r in rows:
        if len(pairs) >= N_POSTMAN_PAIRS:
            break
        path = r.get("path")
        method = r.get("method")
        section = r.get("section", "")
        if not path or not method:
            continue
        action = _action_phrase(section.split("/")[-1], r.get("text"))
        if not action or len(action) < 4:
            continue
        q = random.choice(API_Q_TEMPLATES).format(action=action)
        gold = _api_gold(method, path, None, section.replace("/", " → "), [])
        chunks = context_with_gold(retriever, q, r.get("text"))
        pairs.append(("api_orchestration", r.get("module", "multi"), q, gold, chunks))
    return pairs


# --------------------------------------------------------------------------- #
# Source 2/3 — human-curated dev-benchmark pairs (upweighted)
# --------------------------------------------------------------------------- #
def load_benchmark_pairs(retriever):
    pairs = []
    for l in open(DEV_BENCHMARK):
        d = json.loads(l)
        q = d["question"]
        gold = d["expected_answer"]
        chunks = retriever.retrieve(q, TOP_K)  # authentic inference-time context
        weight = UPWEIGHT.get(d["category"], 1)
        for _ in range(weight):
            pairs.append((d["category"], d["module"], q, gold, chunks))
    return pairs


# --------------------------------------------------------------------------- #
# Grounding guarantee
# --------------------------------------------------------------------------- #
# Matches OHIP-style paths: /rsv/v1/..., /crm/v1/..., with {placeholders}.
PATH_RE = re.compile(r"/[a-zA-Z][\w-]*/v\d+/[\w{}/\-.]*")


def build_path_lookup():
    """path -> stored chunk text, for every endpoint/Postman chunk in the store."""
    lookup = {}
    for fn in (API_CHUNKS, POSTMAN_CHUNKS):
        for l in open(fn):
            r = json.loads(l)
            p, t = r.get("path"), r.get("text")
            if p and t and p not in lookup:
                lookup[p] = t
    return lookup


def enforce_path_grounding(chunks, gold, lookup):
    """Ensure every API path named in the gold answer is visible in the context.

    A grounding dataset must never ask the model to produce a path it cannot see —
    that trains hallucination. Missing gold chunks are injected (displacing the
    lowest-ranked retrieved chunk) so context size stays at TOP_K.
    """
    ctx = "\n".join(c["text"] for c in chunks)
    missing = []
    for path in dict.fromkeys(PATH_RE.findall(gold)):
        path = path.rstrip(".,;:)")
        if path in ctx:
            continue
        text = lookup.get(path)
        if text and text not in [c["text"] for c in chunks]:
            missing.append(
                {"text": text, "meta": {"path": path}, "dist": 0.0, "protected": True}
            )
    if not missing:
        return chunks, True
    keep = chunks[: max(0, TOP_K - len(missing))]
    merged = keep + missing
    random.shuffle(merged)
    new_ctx = "\n".join(c["text"] for c in merged)
    fully = all(
        p.rstrip(".,;:)") in new_ctx for p in dict.fromkeys(PATH_RE.findall(gold))
    )
    return merged, fully


def trim_to_budget(chunks, gold, question):
    """Keep the example inside MAX_CHARS by dropping the *lowest-ranked* retrieved
    chunks — never the answer, and never a chunk that grounds a gold path.

    Truncating from the end would cut the assistant answer off, teaching the model
    to stop mid-sentence; and truncating blindly could remove the very chunk the
    gold path depends on. So the context is what gives.
    """
    budget = MAX_CHARS - len(gold) - len(question) - len(SYSTEM_PROMPT_RAG) - 200
    protected = [c for c in chunks if c.get("protected")]
    optional = [c for c in chunks if not c.get("protected")]
    optional.sort(key=lambda c: c.get("dist", 1.0))  # best first, drop worst first

    kept = list(protected)
    used = sum(len(c["text"]) for c in kept)
    for c in optional:
        if used + len(c["text"]) > budget:
            continue
        kept.append(c)
        used += len(c["text"])

    # A single protected chunk can still blow the budget; hard-truncate its text.
    if used > budget and kept:
        for c in kept:
            if len(c["text"]) > budget // max(1, len(kept)):
                c["text"] = c["text"][: budget // max(1, len(kept))]

    order = {id(c): i for i, c in enumerate(chunks)}
    kept.sort(key=lambda c: order.get(id(c), 999))
    return kept or chunks[:1]


def to_example(category, module, question, gold, chunks):
    user = build_user_prompt(question, chunks)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user", "content": user},
            {"role": "assistant", "content": gold},
        ],
        "meta": {"category": category, "module": module},
    }


def main():
    print("Loading retriever (bge-small + ChromaDB)...")
    retriever = Retriever()

    print("Building API-path grounding pairs...")
    api = load_api_pairs(retriever)
    print(f"  {len(api)} API pairs")

    print("Building Postman workflow pairs...")
    postman = load_postman_pairs(retriever)
    print(f"  {len(postman)} Postman pairs")

    print("Building human-curated dev-benchmark pairs (upweighted)...")
    bench = load_benchmark_pairs(retriever)
    print(f"  {len(bench)} benchmark pairs (after upweighting)")

    all_pairs = api + postman + bench

    print("Enforcing path grounding (every gold path must appear in context)...")
    lookup = build_path_lookup()
    examples, dropped = [], 0
    for category, module, q, gold, chunks in all_pairs:
        chunks, fully = enforce_path_grounding(chunks, gold, lookup)
        if not fully:
            dropped += 1
            continue
        chunks = trim_to_budget(chunks, gold, q)
        examples.append(to_example(category, module, q, gold, chunks))
    print(f"  dropped {dropped} pairs whose gold path could not be grounded")
    random.shuffle(examples)

    n_val = max(1, int(len(examples) * VAL_FRACTION))
    val, train = examples[:n_val], examples[n_val:]

    train_path = DATA_DIR / "sft_train.jsonl"
    val_path = DATA_DIR / "sft_val.jsonl"
    with open(train_path, "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open(val_path, "w") as f:
        for e in val:
            f.write(json.dumps(e) + "\n")

    # stats
    from collections import Counter
    cat = Counter(e["meta"]["category"] for e in examples)
    mod = Counter(e["meta"]["module"] for e in examples)
    lens = [len(e["messages"][1]["content"]) + len(e["messages"][2]["content"]) for e in examples]
    stats = {
        "seed": SEED,
        "total": len(examples),
        "train": len(train),
        "val": len(val),
        "by_category": dict(cat),
        "by_module": dict(mod),
        "sources": {"api": len(api), "postman": len(postman), "benchmark_upweighted": len(bench)},
        "dropped_ungrounded": dropped,
        "char_len_min": min(lens),
        "char_len_median": sorted(lens)[len(lens) // 2],
        "char_len_max": max(lens),
    }
    with open(DATA_DIR / "sft_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== SFT dataset built ===")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
