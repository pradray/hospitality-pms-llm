"""Consolidate every Phase 1/2 result into one markdown file of report-ready tables.

Single source of truth for the dissertation: every number quoted in the report
comes from here, so the prose cannot drift from the data. Re-run it after any new
scoring pass and the tables regenerate.

Usage:
    python src/consolidate_results.py --out docs/final_results.md
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_results import load_valid_paths, path_stats, boot_mean_ci

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "output" / "eval_results"
SEED = 42

# Which scored files belong to which experiment.
SETS = {
    "test_main": ["scored_eval_test_20260809_130422.jsonl",
                  "scored_eval_test_20260809_190402.jsonl"],
    "test_strict": ["scored_eval_test_strict.jsonl"],
    "dev_retrieval": ["scored_dev_retrieval.jsonl"],
    "dev_quant": ["scored_dev_quantisation.jsonl"],
    "dev_k15": ["scored_dev_k15.jsonl"],
    "dev_phase1": ["scored_eval_dev_20260628_183315.jsonl"],
}

LABELS = {
    "3B-base": "Qwen2.5-3B, no retrieval (stock Q4)",
    "3B-base-req": "Qwen2.5-3B, no retrieval (matched Q4)",
    "3B-RAG": "Qwen2.5-3B + RAG (stock Q4)",
    "3B-RAG-req": "Qwen2.5-3B + RAG (matched Q4)",
    "3B-RAG-req-strict": "Qwen2.5-3B + RAG + strict prompt",
    "3B-LoRA": "Qwen2.5-3B QLoRA, no retrieval",
    "3B-LoRA-RAG": "Qwen2.5-3B QLoRA + RAG",
    "3B-LoRA-RAG-strict": "Qwen2.5-3B QLoRA + RAG + strict prompt",
    "3B-LoRA-RAG-rerank": "Qwen2.5-3B QLoRA + RAG + reranker",
    "3B-LoRA-RAG-k15": "Qwen2.5-3B QLoRA + RAG (top-15)",
    "3B-RAG-q8": "Qwen2.5-3B + RAG (Q8)",
    "7B-base": "Qwen2.5-7B, no retrieval",
    "7B-RAG": "Qwen2.5-7B + RAG (stock Q4)",
    "7B-RAG-req": "Qwen2.5-7B + RAG (matched Q4)",
    "7B-RAG-q8": "Qwen2.5-7B + RAG (Q8)",
    "API-ceiling": "grok-4.5, no retrieval",
    "API-ceiling-RAG": "grok-4.5 + RAG",
}


def load(names):
    rows = []
    for n in names:
        p = RESULTS / n
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("score", -1) > 0:
                    rows.append(r)
    return rows


def paired(by, a, b):
    ks = sorted(set(by[a]) & set(by[b]))
    if not ks:
        return None
    x = np.array([by[a][k] for k in ks], float)
    y = np.array([by[b][k] for k in ks], float)
    d = y - x
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    boot = d[idx].mean(axis=1)
    try:
        p = stats.wilcoxon(x, y).pvalue if not np.allclose(d, 0) else 1.0
    except ValueError:
        p = float("nan")
    return dict(n=len(ks), a=x.mean(), b=y.mean(), delta=d.mean(),
                lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=p,
                w=int((d > 0).sum()), l=int((d < 0).sum()), t=int((d == 0).sum()))


def config_table(rows, valid, order=None):
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    rng = np.random.default_rng(SEED)
    out = []
    for cfg in (order or sorted(by)):
        if cfg not in by:
            continue
        rs = by[cfg]
        sc = [r["score"] for r in rs]
        m, (lo, hi) = boot_mean_ci(sc, rng=rng)
        tot = bad = 0
        for r in rs:
            t, b = path_stats(r.get("generated_answer", ""), valid)
            tot += t
            bad += b
        lat = np.array([r.get("latency_seconds", 0) for r in rs], float)
        ch = np.array([len(r.get("generated_answer", "") or "") for r in rs], float)
        out.append(dict(cfg=cfg, mean=m, lo=lo, hi=hi, n=len(sc),
                        bad_pct=(100 * bad / tot if tot else float("nan")),
                        bad=bad, tot=tot,
                        med_lat=np.median(lat), p95=np.percentile(lat, 95),
                        med_chars=np.median(ch),
                        usable=100 * sum(1 for s in sc if s >= 4) / len(sc),
                        wrong=100 * sum(1 for s in sc if s <= 2) / len(sc)))
    return out


def md_config_table(rows_t, title):
    L = [f"**{title}**", "",
         "| Configuration | Mean (1-5) | 95% CI | Score >=4 | Score <=2 | Fabricated paths | Median latency | p95 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows_t:
        L.append(f"| {LABELS.get(r['cfg'], r['cfg'])} | {r['mean']:.2f} | "
                 f"[{r['lo']:.2f}, {r['hi']:.2f}] | {r['usable']:.0f}% | {r['wrong']:.0f}% | "
                 f"{r['bad_pct']:.0f}% ({r['bad']}/{r['tot']}) | {r['med_lat']:.1f}s | {r['p95']:.1f}s |")
    return "\n".join(L) + "\n"


def md_paired(comparisons, title, note=""):
    L = [f"**{title}**", "",
         "| Comparison | Mean A | Mean B | Delta | 95% CI | Wilcoxon p | W/L/T |",
         "|---|---|---|---|---|---|---|"]
    for label, res in comparisons:
        if not res:
            continue
        star = "*" if res["p"] < 0.05 else ""
        L.append(f"| {label} | {res['a']:.2f} | {res['b']:.2f} | {res['delta']:+.2f} | "
                 f"[{res['lo']:+.2f}, {res['hi']:+.2f}] | {res['p']:.4f}{star} | "
                 f"{res['w']}/{res['l']}/{res['t']} |")
    if note:
        L += ["", note]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/final_results.md")
    args = ap.parse_args()

    valid = load_valid_paths()
    S = {k: load(v) for k, v in SETS.items()}
    doc = ["# Consolidated results", "",
           "Generated by `src/consolidate_results.py`. Every figure quoted in the",
           "report is taken from this file.", "",
           f"Corpus attests {len(valid)} distinct API paths.", ""]

    # --- main test matrix ---
    main_rows = S["test_main"] + S["test_strict"]
    order = ["3B-base", "3B-base-req", "3B-LoRA", "7B-base",
             "3B-RAG", "3B-RAG-req", "3B-RAG-req-strict",
             "3B-LoRA-RAG", "3B-LoRA-RAG-strict", "7B-RAG",
             "API-ceiling", "API-ceiling-RAG"]
    doc.append("## Table 1. Held-out test set (46 tasks)")
    doc.append("")
    doc.append(md_config_table(config_table(main_rows, valid, order),
                               "All configurations, 46 held-out tasks, judged by grok-4.3"))

    by = defaultdict(dict)
    for r in main_rows:
        by[r["config"]][r["task_id"]] = r["score"]

    comps = [
        ("Retrieval: 3B base -> 3B + RAG", paired(by, "3B-base-req", "3B-RAG-req")),
        ("Fine-tuning alone: 3B base -> 3B QLoRA", paired(by, "3B-base-req", "3B-LoRA")),
        ("Fine-tuning on RAG: 3B RAG -> 3B QLoRA + RAG", paired(by, "3B-RAG-req", "3B-LoRA-RAG")),
        ("Prompting alone: 3B RAG -> 3B RAG + strict", paired(by, "3B-RAG-req", "3B-RAG-req-strict")),
        ("Both: 3B RAG -> 3B QLoRA + RAG + strict", paired(by, "3B-RAG-req", "3B-LoRA-RAG-strict")),
        ("Capacity: 7B + RAG -> 3B QLoRA + RAG", paired(by, "7B-RAG", "3B-LoRA-RAG")),
        ("Quantisation control: stock Q4 -> matched Q4", paired(by, "3B-base", "3B-base-req")),
        ("Frontier gap: 3B QLoRA + RAG -> grok-4.5", paired(by, "3B-LoRA-RAG", "API-ceiling")),
    ]
    doc.append("## Table 2. Paired comparisons (held-out test set)")
    doc.append("")
    doc.append(md_paired(comps, "Paired by task; delta = second minus first",
                         "* p < 0.05 uncorrected. Bonferroni for 8 planned comparisons: "
                         "alpha = 0.0063."))

    # --- category means and ranking ---
    cats = sorted({r["category"] for r in main_rows})
    doc.append("## Table 3. Mean score by category (held-out test set)")
    doc.append("")
    hdr = "| Configuration | " + " | ".join(c.replace("_", " ") for c in cats) + " |"
    doc += [hdr, "|---" * (len(cats) + 1) + "|"]
    bycfg = defaultdict(list)
    for r in main_rows:
        bycfg[r["config"]].append(r)
    rng = np.random.default_rng(SEED)
    for cfg in order:
        if cfg not in bycfg:
            continue
        cells = []
        for c in cats:
            sc = [r["score"] for r in bycfg[cfg] if r["category"] == c]
            if sc:
                m, (lo, hi) = boot_mean_ci(sc, n_boot=2000, rng=rng)
                cells.append(f"{m:.2f} [{lo:.1f}, {hi:.1f}]")
            else:
                cells.append("-")
        doc.append(f"| {LABELS.get(cfg, cfg)} | " + " | ".join(cells) + " |")
    doc.append("")

    doc.append("## Table 4. Task-type ranking (Friedman, mean rank; 1 = best)")
    doc.append("")
    for c in cats + ["ALL"]:
        tids = sorted({r["task_id"] for r in main_rows if c == "ALL" or r["category"] == c})
        cfgs = [x for x in order if x in by]
        mat = []
        for t in tids:
            row = [by[x].get(t) for x in cfgs]
            if all(v is not None for v in row):
                mat.append(row)
        if len(mat) < 3:
            continue
        arr = np.array(mat, float)
        ranks = np.apply_along_axis(lambda r: stats.rankdata(-r, method="average"), 1, arr)
        mr = ranks.mean(axis=0)
        try:
            fp = stats.friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])]).pvalue
        except ValueError:
            fp = float("nan")
        doc.append(f"**{c}** (n={len(mat)}, Friedman p={fp:.3g})")
        doc.append("")
        doc.append("| Rank | Configuration | Mean rank | Mean score |")
        doc.append("|---|---|---|---|")
        for pos, i in enumerate(np.argsort(mr), 1):
            doc.append(f"| {pos} | {LABELS.get(cfgs[i], cfgs[i])} | {mr[i]:.2f} | {arr[:, i].mean():.2f} |")
        doc.append("")

    # --- ablations on dev ---
    for key, title, pairs in [
        ("dev_retrieval", "Table 5. Reranking (dev, 105 tasks)",
         [("Dense top-5 -> cross-encoder rerank", ("3B-LoRA-RAG", "3B-LoRA-RAG-rerank"))]),
        ("dev_k15", "Table 6. Long context (dev, 46 tasks, 32k window)",
         [("top-5 -> top-15", ("3B-LoRA-RAG", "3B-LoRA-RAG-k15"))]),
        ("dev_quant", "Table 7. Quantisation (dev, 46 tasks)",
         [("3B: Q4 -> Q8", ("3B-RAG-req", "3B-RAG-q8")),
          ("7B: Q4 -> Q8", ("7B-RAG-req", "7B-RAG-q8"))]),
    ]:
        rows = S.get(key) or []
        if not rows:
            continue
        doc.append(f"## {title}")
        doc.append("")
        doc.append(md_config_table(config_table(rows, valid), "Configurations"))
        b2 = defaultdict(dict)
        for r in rows:
            b2[r["config"]][r["task_id"]] = r["score"]
        doc.append(md_paired([(lbl, paired(b2, a, b)) for lbl, (a, b) in pairs
                              if a in b2 and b in b2], "Paired comparison"))

    # --- judge validation ---
    doc.append("## Table 8. Evaluation-judge reliability")
    doc.append("")
    doc.append("| Comparison | n | Exact | Within 1 | kappa (linear) | kappa (quadratic) | Spearman | Mean bias |")
    doc.append("|---|---|---|---|---|---|---|---|")
    for fn, label in [("judge_validation.json", "Domain expert vs grok-4.3"),
                      ("judge_cross_model.json", "grok-4.5 vs grok-4.3")]:
        p = RESULTS / fn
        if not p.exists():
            continue
        d = json.load(open(p))
        bias = d.get("mean_bias", 0)
        doc.append(f"| {label} | {d['n']} | {100*d['exact']:.0f}% | {100*d['within1']:.0f}% | "
                   f"{d['kappa_linear']:.3f} | {d['kappa_quadratic']:.3f} | "
                   f"{d['spearman']:.3f} | {bias:+.2f} |")
    doc.append("")

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(doc))
    print(f"wrote {out}")
    print(f"{len(doc)} lines")


if __name__ == "__main__":
    main()
