"""Objective audit of the benchmark's own reference answers.

Addresses the benchmark-circularity concern: the same person wrote the questions,
the reference answers, and the system under test. Independent human review is the
real fix, but one check needs no reviewer at all — every API path asserted by a
reference answer must actually exist in the public Oracle corpus. Any that does
not is an error in the gold data, and it penalises models for being right.

This produces:
  1. a list of reference answers citing non-existent API paths (bugs in the gold)
  2. a stratified sample for human spot-checking, so a supervisor can review a
     defensible subset rather than all 151 tasks

Usage:
    python src/audit_benchmark.py
    python src/audit_benchmark.py --sample 20 --out docs/benchmark_review_sample.csv
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from analyze_results import extract_paths, normalise_path, load_valid_paths

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED = 42


def audit(tasks, valid):
    bad = []
    total_paths = ok_paths = 0
    for t in tasks:
        paths = [p for p in extract_paths(t.get("expected_answer", "")) if "/v" in p]
        missing = [p for p in paths if normalise_path(p) not in valid]
        total_paths += len(paths)
        ok_paths += len(paths) - len(missing)
        if missing:
            bad.append((t, missing))
    return bad, total_paths, ok_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20,
                    help="Tasks to draw for human review (stratified).")
    ap.add_argument("--out", default="docs/benchmark_review_sample.csv")
    args = ap.parse_args()

    tasks = []
    for split in ("dev", "test"):
        for line in open(DATA_DIR / f"benchmark_{split}.jsonl"):
            d = json.loads(line)
            d["_split"] = split
            tasks.append(d)

    valid = load_valid_paths()
    print(f"{len(tasks)} benchmark tasks | {len(valid)} valid API paths in corpus\n")

    bad, total_paths, ok_paths = audit(tasks, valid)
    print("=" * 78)
    print("REFERENCE-ANSWER AUDIT — API paths asserted by the gold answers")
    print("=" * 78)
    print(f"paths cited by reference answers : {total_paths}")
    print(f"  verified against the corpus    : {ok_paths} ({100*ok_paths/total_paths:.1f}%)")
    print(f"  NOT found in the corpus        : {total_paths-ok_paths} "
          f"({100*(total_paths-ok_paths)/total_paths:.1f}%)")
    print(f"reference answers with >=1 unverifiable path: {len(bad)}/{len(tasks)}\n")

    for t, missing in bad[:15]:
        print(f"  [{t['_split']}] {t['id']} ({t['category']})")
        print(f"      unverifiable: {', '.join(missing)}")
    if len(bad) > 15:
        print(f"  ... and {len(bad)-15} more")

    # Stratified sample for human review
    random.seed(SEED)
    by_stratum = defaultdict(list)
    for t in tasks:
        by_stratum[(t["category"], t.get("difficulty", "?"))].append(t)
    strata = sorted(by_stratum)
    per = max(1, args.sample // len(strata))
    sample = []
    for s in strata:
        sample.extend(random.sample(by_stratum[s], min(per, len(by_stratum[s]))))
    random.shuffle(sample)
    sample = sample[:args.sample]

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "category", "difficulty", "module", "question",
                    "reference_answer",
                    "reviewer_answer_correct_1_5", "reviewer_difficulty_agree_Y_N",
                    "reviewer_comments"])
        for t in sample:
            w.writerow([t["id"], t["_split"], t["category"], t.get("difficulty", ""),
                        t.get("module", ""), t["question"], t.get("expected_answer", ""),
                        "", "", ""])
    print(f"\nStratified review sample ({len(sample)} tasks) -> {out_path}")
    print("Hand this to the supervisor / an independent reviewer: they score whether")
    print("each reference answer is correct and whether the difficulty label is fair.")


if __name__ == "__main__":
    main()
