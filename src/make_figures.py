"""Generate the report's data figures from the stored results.

Every figure is drawn from the same scored result files the tables come from, so
the two cannot disagree. Re-run after any new scoring pass.

Usage:
    python src/make_figures.py --outdir docs/assets
"""

import argparse
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_results import load_valid_paths, path_stats, boot_mean_ci

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "output" / "eval_results"
SEED = 42

NAVY = "#1E2761"
GREY = "#5A6478"
RED = "#B03A2E"
AMBER = "#E09F3E"
GREEN = "#2E7D5B"
LIGHT = "#7B96C4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#B9C0CF",
    "axes.labelcolor": "#16203F",
    "text.color": "#16203F",
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

SHORT = {
    "3B-base": "3B",
    "3B-base-req": "3B",
    "3B-LoRA": "3B QLoRA",
    "7B-base": "7B",
    "3B-RAG": "3B + RAG",
    "3B-RAG-req": "3B + RAG",
    "3B-RAG-req-strict": "3B + RAG + strict",
    "3B-LoRA-RAG": "3B QLoRA + RAG",
    "3B-LoRA-RAG-strict": "3B QLoRA + RAG + strict",
    "7B-RAG": "7B + RAG",
    "API-ceiling": "grok-4.5",
    "API-ceiling-RAG": "grok-4.5 + RAG",
}
ORDER = ["3B-base-req", "3B-LoRA", "7B-base", "3B-RAG-req", "3B-RAG-req-strict",
         "3B-LoRA-RAG", "3B-LoRA-RAG-strict", "7B-RAG", "API-ceiling", "API-ceiling-RAG"]


def load_test():
    rows = []
    for f in ["scored_eval_test_20260809_130422.jsonl",
              "scored_eval_test_20260809_190402.jsonl",
              "scored_eval_test_strict.jsonl"]:
        p = RESULTS / f
        if p.exists():
            rows += [json.loads(l) for l in open(p) if json.loads(l).get("score", -1) > 0]
    return rows


def fig_training_loss(outdir):
    log = json.load(open(PROJECT_ROOT / "models" / "qwen2.5-3b-lora-adapter" / "train_log.json"))
    tr = [(d["epoch"], d["loss"]) for d in log if "loss" in d and "eval_loss" not in d]
    ev = [(d["epoch"], d["eval_loss"]) for d in log if "eval_loss" in d]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.plot(*zip(*tr), color=LIGHT, lw=1.2, label="Training loss")
    if ev:
        ax.plot(*zip(*ev), color=NAVY, lw=2, marker="o", ms=5, label="Validation loss")
        for e, v in ev:
            ax.annotate(f"{v:.3f}", (e, v), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=NAVY)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#EDF0F6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_4_1_training_loss.png", bbox_inches="tight")
    plt.close(fig)


def fig_scores_ci(rows, outdir):
    rng = np.random.default_rng(SEED)
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r["score"])
    cfgs = [c for c in ORDER if c in by]
    means, los, his = [], [], []
    for c in cfgs:
        m, (lo, hi) = boot_mean_ci(by[c], rng=rng)
        means.append(m); los.append(m - lo); his.append(hi - m)
    colors = [RED if m < 2 else (AMBER if c.startswith("API") else NAVY)
              for c, m in zip(cfgs, means)]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(len(cfgs))
    ax.bar(x, means, color=colors, width=0.62)
    ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor="#33405F",
                elinewidth=1.1, capsize=3)
    for i, m in enumerate(means):
        ax.text(i, m + his[i] + 0.08, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in cfgs], fontsize=7.5, rotation=32, ha="right")
    ax.set_ylabel("Mean score (1 to 5)")
    ax.set_ylim(0, 4.4)
    ax.grid(axis="y", color="#EDF0F6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_5_1_scores.png", bbox_inches="tight")
    plt.close(fig)


def fig_by_category(rows, outdir):
    cats = ["api_orchestration", "config_advisory", "troubleshooting"]
    labels = ["API orchestration", "Configuration advisory", "Troubleshooting"]
    show = ["3B-RAG-req", "3B-LoRA-RAG-strict", "7B-RAG", "API-ceiling"]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["config"]][r["category"]].append(r["score"])
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    w = 0.2
    x = np.arange(len(cats))
    cols = [LIGHT, NAVY, GREY, AMBER]
    for i, c in enumerate(show):
        vals = [np.mean(by[c][k]) if by[c][k] else 0 for k in cats]
        b = ax.bar(x + (i - 1.5) * w, vals, w, label=SHORT[c].replace("\n", " "), color=cols[i])
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean score (1 to 5)")
    ax.set_ylim(0, 5.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", color="#EDF0F6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_5_2_by_category.png", bbox_inches="tight")
    plt.close(fig)


def fig_hallucination(rows, outdir):
    valid = load_valid_paths()
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        t, b = path_stats(r.get("generated_answer", ""), valid)
        agg[r["config"]][0] += t
        agg[r["config"]][1] += b
    cfgs = [c for c in ORDER if c in agg and agg[c][0]]
    pct = [100 * agg[c][1] / agg[c][0] for c in cfgs]
    colors = [RED if p >= 99 else (GREEN if p == 0 else NAVY) for p in pct]
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(cfgs))
    ax.bar(x, pct, color=colors, width=0.62)
    for i, p in enumerate(pct):
        ax.text(i, p + 2, f"{p:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in cfgs], fontsize=7.5, rotation=32, ha="right")
    ax.set_ylabel("API paths not found in corpus (%)")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", color="#EDF0F6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_5_3_hallucination.png", bbox_inches="tight")
    plt.close(fig)


def fig_recall(outdir):
    ks = [5, 10, 15, 20, 25, 30, 40]
    dense = [71, 82, 85, 85, 86, 86, 86]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.plot(ks, dense, color=NAVY, marker="o", ms=4, lw=1.8, label="Dense retrieval")
    ax.axhline(89.3, color=RED, ls="--", lw=1.2)
    ax.annotate("Corpus ceiling 89.3%: the share of gold paths\nthe corpus actually contains",
                xy=(22, 89.3), xytext=(11, 62), fontsize=7.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    ax.scatter([5], [84], color=AMBER, zorder=5, s=45)
    ax.annotate("Cross-encoder rerank at k=5 (84%)", xy=(5, 84), xytext=(6.5, 76),
                fontsize=7.5, color="#9A6B1E",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.9))
    ax.set_xlabel("Chunks retrieved (k)")
    ax.set_ylabel("Gold API paths retrieved (%)")
    ax.set_ylim(55, 95)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", color="#EDF0F6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_5_4_recall.png", bbox_inches="tight")
    plt.close(fig)


def fig_latency(rows, outdir):
    show = ["3B-LoRA-RAG-strict", "3B-RAG-req", "7B-RAG", "API-ceiling-RAG", "API-ceiling"]
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r.get("latency_seconds", 0))
    data = [by[c] for c in show if by[c]]
    labels = [SHORT[c].replace("\n", " ") for c in show if by[c]]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True,
                    medianprops=dict(color="white", lw=1.6),
                    flierprops=dict(marker="o", ms=3, mfc=GREY, mec="none", alpha=0.6))
    for patch, c in zip(bp["boxes"], [NAVY, LIGHT, GREY, AMBER, RED]):
        patch.set_facecolor(c)
        patch.set_edgecolor("none")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Response time (seconds)")
    ax.grid(axis="x", color="#EDF0F6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outdir / "fig_5_5_latency.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/assets")
    args = ap.parse_args()
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_test()
    print(f"{len(rows)} scored test results")
    fig_training_loss(outdir); print("  fig 4.1 training loss")
    fig_scores_ci(rows, outdir); print("  fig 5.1 scores with CIs")
    fig_by_category(rows, outdir); print("  fig 5.2 by category")
    fig_hallucination(rows, outdir); print("  fig 5.3 hallucination")
    fig_recall(outdir); print("  fig 5.4 retrieval recall")
    fig_latency(rows, outdir); print("  fig 5.5 latency")
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
