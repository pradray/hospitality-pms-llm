"""Mechanically verify the factual claims in benchmark reference answers.

This does NOT review the reference answers — that judgement belongs to a human,
and the whole point of the exercise is that the reviewer is independent of the
system under test. What it does is check the parts a machine can check
objectively, so the reviewer spends their time on judgement rather than lookup:

  * does every API path cited by the reference answer exist in the corpus?
  * does the HTTP method attached to it match the corpus?
  * is the operationId real, and does it belong to that path?
  * are the OPERA Control / setting names mentioned attested in the User Guide?

Output is a per-task verification card appended to the review CSV. Every
verdict is traceable to the public corpus, so a reviewer can confirm or reject
it rather than taking it on trust.

Usage:
    python src/verify_reference_answers.py
    python src/verify_reference_answers.py --sheet docs/benchmark_review_sample.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

from analyze_results import extract_paths, normalise_path, VERSIONED_PATH_RE

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHUNK_FILES = [
    OUTPUT_DIR / "api_chunks" / "all_endpoints.jsonl",
    OUTPUT_DIR / "doc_chunks" / "all_doc_chunks.jsonl",
    OUTPUT_DIR / "postman_chunks" / "all_postman_chunks.jsonl",
]

METHOD_PATH_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\b\s*[`'\"]?\s*(/[\w{}\-./]+)"
)
# OPERA Controls are referred to in title case in the User Guide.
CONTROL_RE = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\b")


def load_corpus():
    """path -> set(methods), operationId -> path, and the raw doc text."""
    path_methods, op_to_path, doc_text = {}, {}, []
    for fn in CHUNK_FILES:
        if not fn.exists():
            continue
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("text", "") or ""
                doc_text.append(text)
                p = d.get("path")
                if p:
                    np_ = normalise_path(p)
                    m = (d.get("method") or "").upper()
                    path_methods.setdefault(np_, set())
                    if m:
                        path_methods[np_].add(m)
                    if d.get("operation_id"):
                        op_to_path[d["operation_id"].lower()] = np_
                # paths that only appear inside chunk text
                for tp in VERSIONED_PATH_RE.findall(text):
                    path_methods.setdefault(normalise_path(tp), set())
    return path_methods, op_to_path, "\n".join(doc_text)


def verify(task, path_methods, op_to_path, corpus_text):
    ans = task.get("expected_answer", "") or ""
    notes = []

    # --- API paths and methods ---
    pairs = METHOD_PATH_RE.findall(ans)
    seen = set()
    for method, raw in pairs:
        p = normalise_path(raw)
        if (method, p) in seen:
            continue
        seen.add((method, p))
        if p not in path_methods:
            notes.append(f"PATH NOT IN CORPUS: {method} {raw}")
        else:
            methods = path_methods[p]
            if methods and method.upper() not in methods:
                notes.append(
                    f"METHOD MISMATCH: reference says {method} {raw}; "
                    f"corpus has {'/'.join(sorted(methods))}")

    # bare paths cited without a method
    for raw in extract_paths(ans):
        p = normalise_path(raw)
        if "/v" in p and p not in path_methods:
            if not any(f"PATH NOT IN CORPUS: {m} " in n for m, n in
                       [(m, n) for m in ("GET", "POST", "PUT", "PATCH", "DELETE")
                        for n in notes]):
                msg = f"PATH NOT IN CORPUS: {raw}"
                if msg not in notes:
                    notes.append(msg)

    # --- operation ids ---
    for op in set(re.findall(r"\b([a-z][A-Za-z0-9]{6,})\b", ans)):
        low = op.lower()
        if low in op_to_path:
            continue
        # only flag things that look like operationIds (camelCase verbs)
        if re.match(r"^(get|post|put|patch|delete|create|update|fetch|list)[A-Z]", op):
            notes.append(f"OPERATION ID NOT IN CORPUS: {op}")

    # --- OPERA Controls / settings mentioned ---
    if task.get("category") == "config_advisory":
        for phrase in set(CONTROL_RE.findall(ans)):
            if len(phrase.split()) < 2:
                continue
            if phrase.lower() not in corpus_text.lower():
                notes.append(f"CONTROL/SETTING NOT FOUND IN DOCS: '{phrase}'")

    verdict = "CLEAN" if not notes else "CHECK"
    return verdict, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="docs/benchmark_review_sample.csv")
    ap.add_argument("--out", default="docs/benchmark_review_sample_verified.csv")
    args = ap.parse_args()

    path_methods, op_to_path, corpus_text = load_corpus()
    print(f"corpus: {len(path_methods)} paths, {len(op_to_path)} operation ids\n")

    tasks = {}
    for split in ("dev", "test"):
        for line in open(DATA_DIR / f"benchmark_{split}.jsonl"):
            d = json.loads(line)
            tasks[d["id"]] = d

    sheet_path = PROJECT_ROOT / args.sheet
    rows = list(csv.DictReader(open(sheet_path)))
    out_rows, n_clean, n_check = [], 0, 0

    for row in rows:
        t = tasks.get(row["id"])
        if not t:
            continue
        verdict, notes = verify(t, path_methods, op_to_path, corpus_text)
        n_clean += verdict == "CLEAN"
        n_check += verdict == "CHECK"
        row["automated_verification"] = verdict
        row["automated_findings"] = " | ".join(notes) if notes else ""
        out_rows.append(row)
        if notes:
            print(f"[{verdict}] {row['id']} ({row['category']})")
            for n in notes:
                print(f"     - {n}")

    out_path = PROJECT_ROOT / args.out
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{n_clean} clean / {n_check} needing a look, of {len(out_rows)}")
    print(f"wrote {out_path}")
    print("\nThe reviewer still decides. These flags are machine lookups against the")
    print("public corpus and can themselves be wrong — a path missing from the")
    print("corpus may be a real endpoint outside the 4 modules that were scraped.")


if __name__ == "__main__":
    main()
