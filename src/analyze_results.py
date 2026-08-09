"""Statistical analysis and error taxonomy over scored evaluation results.

The existing summary in score_results.py is purely descriptive (means by config,
category, module). For the dissertation we need to say whether a difference
between two configs is real, and *why* answers fail. This adds:

  1. Bootstrap 95% CIs on each config's mean score.
  2. PAIRED comparisons between configs. Every config answers the same benchmark
     tasks, so pairing by task_id removes task difficulty as a source of variance
     — an unpaired test on 46 tasks would be needlessly weak. Reports a paired
     bootstrap CI on the mean difference plus a Wilcoxon signed-rank test (the
     scores are 1-5 ordinal, so a t-test's normality assumption is unjustified).
  3. Hallucinated-API-path rate, measured MECHANICALLY rather than via the judge:
     every API path in an answer is checked against the set of real paths in the
     corpus. Phase 1 attributed 48% of errors to hallucinated paths, and this is
     the metric the fine-tune was built to move, so it should not depend on the
     judge's opinion.

Usage:
    python src/analyze_results.py --results output/eval_results/scored_eval_test_*.jsonl
    python src/analyze_results.py --results <file> --baseline 3B-RAG-req --against 3B-LoRA-RAG
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
API_CHUNKS = OUTPUT_DIR / "api_chunks" / "all_endpoints.jsonl"
POSTMAN_CHUNKS = OUTPUT_DIR / "postman_chunks" / "all_postman_chunks.jsonl"

SEED = 42
N_BOOT = 10000

# Two ways to spot a cited endpoint. Matching only the well-formed OHIP shape
# (/crm/v1/...) is a trap: a model that invents "/v1/guest/{guestId}/certificates"
# — no module prefix, not a real OHIP path — would score as citing no paths at
# all. That systematically hides the most badly hallucinated answers, so anything
# announced with an HTTP method counts as a cited path too.
METHOD_PATH_RE = re.compile(
    r"\b(?:GET|POST|PUT|PATCH|DELETE)\b\s*[`'\"]?\s*(/[\w{}\-./]+)"
)
VERSIONED_PATH_RE = re.compile(r"/[\w{}\-.]+/v\d+/[\w{}\-./]*")


def extract_paths(text: str) -> list[str]:
    found = METHOD_PATH_RE.findall(text or "") + VERSIONED_PATH_RE.findall(text or "")
    out = []
    for p in found:
        p = normalise_path(p)
        # Require at least two segments and a letter, so prose fragments and bare
        # numbers ("/2", "and/or") are not mistaken for endpoints.
        if p.count("/") >= 2 and re.search(r"[a-z]", p) and p not in out:
            out.append(p)
    return out


def normalise_path(path: str) -> str:
    """Canonical form for comparing paths.

    Placeholder *names* are not what we are judging — a model writing
    /rsv/v1/hotels/{hotelId}/reservations instead of {{HotelId}} is correct.
    Collapse every placeholder to {} and lowercase, so only real structural
    differences count as hallucinations.
    """
    p = path.rstrip(".,;:)'\"`").rstrip("/")
    p = re.sub(r"\{\{[^}]*\}\}", "{}", p)
    p = re.sub(r"\{[^}]*\}", "{}", p)
    return p.lower()


DOC_CHUNKS = OUTPUT_DIR / "doc_chunks" / "all_doc_chunks.jsonl"


def load_valid_paths() -> set[str]:
    """Every API path the corpus actually attests.

    Reading only the `path` metadata field misses paths that appear inside chunk
    *text* — Postman workflow steps and User Guide examples cite endpoints that
    are never a chunk's own path (1,874 metadata paths vs 2,725 once text is
    included). Using the smaller set marks genuine endpoints as hallucinated, so
    the union is the honest reference for "does this endpoint exist".
    """
    valid = set()
    for fn in (API_CHUNKS, POSTMAN_CHUNKS, DOC_CHUNKS):
        if not fn.exists():
            continue
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                if d.get("path"):
                    valid.add(normalise_path(d["path"]))
                for p in VERSIONED_PATH_RE.findall(d.get("text", "") or ""):
                    valid.add(normalise_path(p))
    return valid


def path_stats(answer: str, valid: set[str]) -> tuple[int, int]:
    """(distinct paths cited, those that do not exist in the corpus)."""
    paths = extract_paths(answer)
    if not paths:
        return 0, 0
    bad = sum(1 for p in paths if p not in valid)
    return len(paths), bad


def boot_mean_ci(values, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(SEED)
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    means = a[idx].mean(axis=1)
    return a.mean(), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_compare(scores_a: dict, scores_b: dict, label_a: str, label_b: str, rng):
    """Paired bootstrap + Wilcoxon on tasks answered by both configs."""
    common = sorted(set(scores_a) & set(scores_b))
    if not common:
        return None
    a = np.array([scores_a[t] for t in common], dtype=float)
    b = np.array([scores_b[t] for t in common], dtype=float)
    diff = b - a

    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    if np.allclose(diff, 0):
        p = 1.0
    else:
        try:
            p = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
        except ValueError:
            p = float("nan")

    return {
        "n": len(common),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "delta": diff.mean(),
        "ci": (float(lo), float(hi)),
        "p": float(p),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
        "label_a": label_a,
        "label_b": label_b,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, nargs="+")
    ap.add_argument("--baseline", help="Config to compare everything against.")
    ap.add_argument("--against", nargs="*", help="Specific configs to compare to --baseline.")
    ap.add_argument("--json-out", help="Write the computed numbers to this path.")
    args = ap.parse_args()

    rows = []
    for pattern in args.results:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if not Path(path).exists():
                continue
            with open(path) as f:
                for line in f:
                    rows.append(json.loads(line))
    rows = [r for r in rows if r.get("score", -1) > 0]
    if not rows:
        raise SystemExit("No scored rows found.")

    valid = load_valid_paths()
    print(f"Loaded {len(rows)} scored results | {len(valid)} valid API paths in corpus\n")

    rng = np.random.default_rng(SEED)
    by_config = defaultdict(list)
    scores_by_config = defaultdict(dict)
    for r in rows:
        by_config[r["config"]].append(r)
        scores_by_config[r["config"]][r["task_id"]] = r["score"]

    # ---- per-config means with bootstrap CIs + hallucination rate ----
    print("=" * 92)
    print("PER-CONFIG SCORES (bootstrap 95% CI) AND HALLUCINATED API PATHS")
    print("=" * 92)
    header = (f"{'Config':<16}{'Mean':>6}{'95% CI':>18}{'N':>5}"
              f"{'Ans w/ bad path':>17}{'Bad/all paths':>16}{'Latency':>9}")
    print(header)
    print("-" * 92)

    summary = {}
    for cfg in sorted(by_config):
        rs = by_config[cfg]
        scores = [r["score"] for r in rs]
        mean, (lo, hi) = boot_mean_ci(scores, rng=rng)

        ans_with_bad = tot_paths = bad_paths = ans_with_paths = 0
        for r in rs:
            t, b = path_stats(r.get("generated_answer", ""), valid)
            tot_paths += t
            bad_paths += b
            if t:
                ans_with_paths += 1
                if b:
                    ans_with_bad += 1
        pct_ans = 100 * ans_with_bad / ans_with_paths if ans_with_paths else float("nan")
        pct_path = 100 * bad_paths / tot_paths if tot_paths else float("nan")
        lat = np.mean([r.get("latency_seconds", 0) for r in rs])

        print(f"{cfg:<16}{mean:>6.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}{len(scores):>5}"
              f"{f'{pct_ans:.0f}% ({ans_with_bad}/{ans_with_paths})':>17}"
              f"{f'{pct_path:.0f}% ({bad_paths}/{tot_paths})':>16}{lat:>8.1f}s")

        summary[cfg] = {
            "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(scores),
            "answers_with_bad_path_pct": pct_ans, "bad_path_pct": pct_path,
            "bad_paths": bad_paths, "total_paths": tot_paths,
            "mean_latency_s": float(lat),
        }

    # ---- per-config × category, WITH intervals ----
    # Breakdowns cut 46 tasks into cells of ~13-17. A bare mean there invites
    # over-reading; the CI shows how little these cells actually pin down.
    print("\n" + "=" * 92)
    print("MEAN SCORE BY CONFIG x CATEGORY (bootstrap 95% CI)")
    print("=" * 92)
    cats = sorted({r["category"] for r in rows})
    print(f"{'Config':<16}" + "".join(f"{c:>25}" for c in cats))
    print("-" * 92)
    for cfg in sorted(by_config):
        line = f"{cfg:<16}"
        for c in cats:
            sc = [r["score"] for r in by_config[cfg] if r["category"] == c]
            if sc:
                m, (lo, hi) = boot_mean_ci(sc, n_boot=2000, rng=rng)
                line += f"{f'{m:.2f} [{lo:.1f},{hi:.1f}] n={len(sc)}':>25}"
            else:
                line += f"{'-':>25}"
        print(line)

    # ---- ranking across configs, per task type ----
    # Which system is best depends on the task type, so a single overall mean is
    # misleading. Friedman tests whether the configs differ at all within a
    # category (non-parametric, paired across tasks); mean rank orders them.
    print("\n" + "=" * 92)
    print("TASK-TYPE RANKING (mean rank across tasks; 1 = best)")
    print("=" * 92)
    ranked_cfgs = sorted(by_config)
    for c in cats + ["ALL"]:
        task_ids = sorted({r["task_id"] for r in rows
                           if c == "ALL" or r["category"] == c})
        mat = []
        for t in task_ids:
            row = []
            for cfg in ranked_cfgs:
                s = scores_by_config[cfg].get(t)
                if s is None:
                    row = None
                    break
                row.append(s)
            if row:
                mat.append(row)
        if len(mat) < 3:
            continue
        arr = np.array(mat, dtype=float)
        # average ranks, high score = rank 1
        ranks = np.apply_along_axis(
            lambda r: stats.rankdata(-r, method="average"), 1, arr)
        mean_ranks = ranks.mean(axis=0)
        try:
            fp = stats.friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])]).pvalue
        except ValueError:
            fp = float("nan")
        order = np.argsort(mean_ranks)
        print(f"\n  {c}  (n={len(mat)} tasks, Friedman p={fp:.4g})")
        for pos, i in enumerate(order, 1):
            print(f"    {pos:>2}. {ranked_cfgs[i]:<18} mean rank {mean_ranks[i]:.2f}"
                  f"   mean score {arr[:, i].mean():.2f}")

    # ---- latency distribution ----
    # Means hide the tail; a p95 of 60s is a different product from a mean of 8s.
    print("\n" + "=" * 92)
    print("LATENCY DISTRIBUTION (seconds) — and answer length, which drives it")
    print("=" * 92)
    print(f"{'Config':<16}{'median':>8}{'p90':>8}{'p95':>8}{'max':>8}{'mean':>8}"
          f"{'chars(med)':>12}{'ms/char':>9}")
    print("-" * 92)
    for cfg in sorted(by_config):
        lat = np.array([r.get("latency_seconds", 0) for r in by_config[cfg]], dtype=float)
        chars = np.array([len(r.get("generated_answer", "") or "")
                          for r in by_config[cfg]], dtype=float)
        mspc = 1000 * lat.sum() / chars.sum() if chars.sum() else float("nan")
        print(f"{cfg:<16}{np.median(lat):>8.1f}{np.percentile(lat,90):>8.1f}"
              f"{np.percentile(lat,95):>8.1f}{lat.max():>8.1f}{lat.mean():>8.1f}"
              f"{np.median(chars):>12.0f}{mspc:>9.1f}")
    print("\n  ms/char normalises for verbosity: a config can look 'fast' purely")
    print("  because it emits shorter answers.")

    # ---- paired comparisons ----
    comparisons = []
    if args.baseline:
        targets = args.against or [c for c in sorted(by_config) if c != args.baseline]
        for t in targets:
            if t in scores_by_config and args.baseline in scores_by_config:
                comparisons.append((args.baseline, t))
    else:
        # Default: the comparisons this dissertation actually argues about.
        defaults = [
            ("3B-base-req", "3B-LoRA"),        # fine-tuning alone, matched quantisation
            ("3B-RAG-req", "3B-LoRA-RAG"),     # fine-tuning on top of RAG (the key one)
            ("3B-base-req", "3B-RAG-req"),     # retrieval alone
            ("3B-RAG", "3B-LoRA-RAG"),         # vs the Phase 1 RAG baseline
            ("7B-RAG", "3B-LoRA-RAG"),         # can a tuned 3B match a 7B?
            ("3B-base", "3B-base-req"),        # quantisation-recipe control
        ]
        comparisons = [(a, b) for a, b in defaults
                       if a in scores_by_config and b in scores_by_config]

    if comparisons:
        print("\n" + "=" * 92)
        print("PAIRED COMPARISONS  (delta = second - first; paired bootstrap CI, Wilcoxon p)")
        print("=" * 92)
        print(f"{'Comparison':<34}{'delta':>8}{'95% CI':>18}{'p':>10}{'W/L/T':>14}")
        print("-" * 92)
        results_json = []
        for a, b in comparisons:
            res = paired_compare(scores_by_config[a], scores_by_config[b], a, b, rng)
            if not res:
                continue
            sig = "*" if res["p"] < 0.05 else " "
            name = f"{a} -> {b}"
            print(f"{name:<34}{res['delta']:>+8.2f}"
                  f"{f'[{res['ci'][0]:+.2f}, {res['ci'][1]:+.2f}]':>18}"
                  f"{res['p']:>9.4f}{sig}"
                  f"{f'{res['wins']}/{res['losses']}/{res['ties']}':>14}")
            results_json.append(res)
        print("\n* p < 0.05 (Wilcoxon signed-rank, paired by task).")
        print("A CI that crosses 0 means the difference is not distinguishable from noise.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"per_config": summary, "comparisons": results_json if comparisons else []},
                      f, indent=2, default=float)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
