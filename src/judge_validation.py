"""Validate the LLM judge against human scoring.

Every result in this dissertation rests on scores produced by an LLM judge. If
the judge does not agree with a domain expert, none of the comparisons mean much.
This builds a blind annotation sheet and, once it is filled in, computes
agreement.

Blind by construction: the sheet carries the question, the reference answer and
the generated answer, but NOT the judge's score, and rows are shuffled so
config identity cannot be inferred from ordering. Seeing the machine score first
is the classic way to anchor human ratings and inflate agreement.

Agreement is reported three ways, because they answer different questions:
  * exact-match accuracy      - blunt, and harsh on a 1-5 scale
  * Cohen's kappa (linear and quadratic weights) - chance-corrected, and weighted
    so that a 4-vs-5 disagreement counts less than 1-vs-5
  * Spearman correlation      - does the judge RANK answers the way a human does,
    which is what the config comparisons actually depend on

Usage:
    # 1. make the sheet
    python src/judge_validation.py --make-sheet --n 20
    # 2. fill in human_score in docs/judge_validation_sheet.csv
    # 3. compute agreement
    python src/judge_validation.py --score
"""

import argparse
import csv
import json
import glob
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "eval_results"
SHEET = PROJECT_ROOT / "docs" / "judge_validation_sheet.csv"
KEY = PROJECT_ROOT / "output" / "eval_results" / "judge_validation_key.json"
SEED = 42


def make_sheet(n: int, pattern: str):
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("score", -1) > 0:
                    rows.append(r)
    if not rows:
        raise SystemExit(f"No scored results matched {pattern}")

    # Stratify across category AND judge score, so the sample spans the whole
    # scale. Sampling at random would over-represent whatever score dominates
    # and leave agreement undefined at the ends.
    random.seed(SEED)
    buckets = {}
    for r in rows:
        buckets.setdefault((r["category"], r["score"]), []).append(r)
    keys = sorted(buckets)
    # Ceiling, not floor: with 15 strata and n=24, floor gives 1 per stratum and
    # silently returns 15 items instead of the 24 asked for.
    per = max(1, -(-n // len(keys)))
    sample = []
    for k in keys:
        sample.extend(random.sample(buckets[k], min(per, len(buckets[k]))))
    random.shuffle(sample)
    sample = sample[:n]

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "category", "question", "reference_answer",
                    "generated_answer", "human_score_1_5", "human_notes"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["category"], r["question"], r["expected_answer"],
                        r["generated_answer"], "", ""])

    with open(KEY, "w") as f:
        json.dump([{"item": i, "task_id": r["task_id"], "config": r["config"],
                    "judge_score": r["score"], "judge_reason": r.get("score_reason", "")}
                   for i, r in enumerate(sample, 1)], f, indent=2)

    print(f"Wrote {len(sample)} items -> {SHEET}")
    print(f"Judge scores held back in    -> {KEY}")
    print("\nScore each generated answer 1-5 against the SAME rubric the judge used:")
    print("  5 excellent / 4 good / 3 adequate / 2 poor / 1 wrong")
    print("Do not open the key file until the sheet is complete.")


def score():
    import numpy as np
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score

    key = {k["item"]: k for k in json.load(open(KEY))}
    human, judge, cats = [], [], []
    with open(SHEET) as f:
        for row in csv.DictReader(f):
            v = (row.get("human_score_1_5") or "").strip()
            if not v:
                continue
            item = int(row["item"])
            human.append(int(float(v)))
            judge.append(key[item]["judge_score"])
            cats.append(row["category"])

    if len(human) < 5:
        raise SystemExit(f"Only {len(human)} rows scored — fill in more of {SHEET}")

    h, j = np.array(human), np.array(judge)
    exact = float((h == j).mean())
    within1 = float((np.abs(h - j) <= 1).mean())
    k_lin = cohen_kappa_score(h, j, weights="linear")
    k_quad = cohen_kappa_score(h, j, weights="quadratic")
    rho, prho = stats.spearmanr(h, j)
    bias = float((j - h).mean())

    print(f"=== JUDGE VALIDATION (n={len(h)}) ===")
    print(f"  exact agreement      : {100*exact:.0f}%")
    print(f"  within 1 point       : {100*within1:.0f}%")
    print(f"  Cohen kappa (linear) : {k_lin:.3f}")
    print(f"  Cohen kappa (quad)   : {k_quad:.3f}")
    print(f"  Spearman rho         : {rho:.3f} (p={prho:.4f})")
    print(f"  mean judge - human   : {bias:+.2f} "
          f"({'judge scores higher' if bias > 0 else 'judge scores lower'})")
    print("\n  Landis & Koch: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,")
    print("                 0.61-0.80 substantial, >0.80 almost perfect")
    print("\n  Report the WEIGHTED kappa: the scale is ordinal, so an unweighted")
    print("  kappa would treat 4-vs-5 as badly as 1-vs-5.")

    out = RESULTS_DIR / "judge_validation.json"
    with open(out, "w") as f:
        json.dump({"n": len(h), "exact": exact, "within1": within1,
                   "kappa_linear": float(k_lin), "kappa_quadratic": float(k_quad),
                   "spearman": float(rho), "spearman_p": float(prho),
                   "mean_bias": bias}, f, indent=2)
    print(f"\nwrote {out}")


def cross_judge(n: int, pattern: str, model_b: str):
    """Re-score a sample with a SECOND judge model and measure agreement.

    This is a robustness check, NOT a substitute for human validation: two LLMs
    agreeing tells you the scores are stable, not that they are correct, and both
    judges available here are Grok models, so shared blind spots are likely.
    Report it as inter-judge reliability and keep it clearly separate from any
    human-agreement number.
    """
    import numpy as np
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score

    sys_path = Path(__file__).parent
    import sys
    sys.path.insert(0, str(sys_path))
    from score_results import _load_env, score_with_grok, JUDGE_PROMPT

    _load_env()

    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("score", -1) > 0:
                    rows.append(r)
    random.seed(SEED)
    random.shuffle(rows)
    sample = rows[:n]
    print(f"Re-scoring {len(sample)} results with {model_b} ...")

    a_scores, b_scores = [], []
    for i, r in enumerate(sample, 1):
        prompt = JUDGE_PROMPT.format(
            category=r["category"], difficulty=r["difficulty"], module=r["module"],
            question=r["question"], expected_answer=r["expected_answer"],
            generated_answer=r["generated_answer"],
        )
        try:
            v = score_with_grok(prompt, model_b)
            b_scores.append(int(v["score"]))
            a_scores.append(int(r["score"]))
        except Exception as e:
            print(f"  [{i}] failed: {type(e).__name__}")
        if i % 20 == 0:
            print(f"  {i}/{len(sample)}")

    a, b = np.array(a_scores), np.array(b_scores)
    k_lin = cohen_kappa_score(a, b, weights="linear")
    k_quad = cohen_kappa_score(a, b, weights="quadratic")
    rho, prho = stats.spearmanr(a, b)
    print(f"\n=== INTER-JUDGE AGREEMENT (n={len(a)}) ===")
    print(f"  judge A (primary)    : grok-4.3")
    print(f"  judge B (secondary)  : {model_b}")
    print(f"  exact agreement      : {100*(a==b).mean():.0f}%")
    print(f"  within 1 point       : {100*(np.abs(a-b)<=1).mean():.0f}%")
    print(f"  Cohen kappa (linear) : {k_lin:.3f}")
    print(f"  Cohen kappa (quad)   : {k_quad:.3f}")
    print(f"  Spearman rho         : {rho:.3f} (p={prho:.4g})")
    print(f"  mean B - A           : {(b-a).mean():+.2f}")
    print("\n  NOTE: both judges are Grok models. This measures stability, not")
    print("  correctness, and does not replace expert human validation.")

    out = RESULTS_DIR / "judge_cross_model.json"
    with open(out, "w") as f:
        json.dump({"n": len(a), "judge_a": "grok-4.3", "judge_b": model_b,
                   "exact": float((a == b).mean()),
                   "within1": float((np.abs(a-b) <= 1).mean()),
                   "kappa_linear": float(k_lin), "kappa_quadratic": float(k_quad),
                   "spearman": float(rho), "mean_bias": float((b-a).mean())}, f, indent=2)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sheet", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cross-judge", action="store_true",
                    help="Inter-judge reliability with a second model.")
    ap.add_argument("--model-b", default="grok-4.5")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pattern",
                    default=str(RESULTS_DIR / "scored_eval_test_*.jsonl"))
    args = ap.parse_args()
    if args.make_sheet:
        make_sheet(args.n, args.pattern)
    elif args.score:
        score()
    elif args.cross_judge:
        cross_judge(args.n, args.pattern, args.model_b)
    else:
        ap.error("pass --make-sheet, --score or --cross-judge")


if __name__ == "__main__":
    main()
