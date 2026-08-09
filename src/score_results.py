"""Score evaluation results using LLM-as-judge.

Takes eval_*.jsonl files, scores each generated answer against the
expected answer on a 1-5 scale using Claude as judge, then outputs
scored results and summary statistics.

Usage:
    python src/score_results.py --results output/eval_results/eval_dev_*.jsonl
    python src/score_results.py --results output/eval_results/eval_dev_*.jsonl --judge openai
"""

import argparse
import json
import os
import time
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "eval_results"

JUDGE_PROMPT = """\
You are an expert evaluator for a hospitality PMS (Property Management System) domain knowledge benchmark.

Score the GENERATED ANSWER against the EXPECTED ANSWER on a 1-5 scale:

5 = Excellent: Covers all key points, correct API paths/operations, technically accurate
4 = Good: Covers most key points, minor omissions or imprecisions
3 = Adequate: Covers some key points but misses important details
2 = Poor: Mostly incorrect or misses the main point, some relevant content
1 = Wrong: Completely incorrect, irrelevant, or refuses to answer

Scoring guidelines:
- For API orchestration tasks: correct HTTP methods and paths are critical. Wrong method or path = max score 2.
- For config advisory tasks: naming the correct OPERA Control or setting is critical. Generic advice without specifics = max score 3.
- For troubleshooting tasks: identifying the root cause is critical. Listing generic troubleshooting steps without identifying the specific issue = max score 3.
- Partial credit: if the answer gets the main point right but misses secondary details, score 3-4.
- Extra correct information beyond the expected answer should not reduce the score.
- Hallucinated API paths or OPERA Control names that don't exist should reduce score by 1-2 points.

TASK CATEGORY: {category}
TASK DIFFICULTY: {difficulty}
TASK MODULE: {module}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <1-5>, "reason": "<one sentence justification>"}}"""


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def _parse_judge_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def score_with_anthropic(prompt: str, model: str | None = None) -> dict:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model or "claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_json(response.content[0].text)


def score_with_openai(prompt: str, model: str | None = None) -> dict:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_json(response.choices[0].message.content)


# Phase 1 judged with "grok-3-mini", which xAI has since retired — the endpoint
# no longer serves it. Judge model is now selectable via --judge-model.
# NOTE for the write-up: because the judge model changed between Phase 1 (dev)
# and Phase 2 (test), scores must not be compared *across* those runs. All
# comparisons in the final results are within a single scoring run.
GROK_JUDGE_MODEL = "grok-4.3"


def score_with_grok(prompt: str, model: str | None = None) -> dict:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    response = client.chat.completions.create(
        model=model or GROK_JUDGE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_judge_json(response.choices[0].message.content)


def score_result(result: dict, judge_fn) -> dict:
    prompt = JUDGE_PROMPT.format(
        category=result["category"],
        difficulty=result["difficulty"],
        module=result["module"],
        question=result["question"],
        expected_answer=result["expected_answer"],
        generated_answer=result["generated_answer"],
    )

    try:
        verdict = judge_fn(prompt)
        result["score"] = verdict["score"]
        result["score_reason"] = verdict["reason"]
    except Exception as e:
        result["score"] = -1
        result["score_reason"] = f"Scoring error: {type(e).__name__}: {e}"

    return result


def print_summary(results: list[dict]):
    scored = [r for r in results if r.get("score", -1) > 0]
    if not scored:
        print("No scored results.")
        return

    # Overall
    avg = sum(r["score"] for r in scored) / len(scored)
    print(f"\nOverall: {avg:.2f} avg score ({len(scored)} tasks scored)")

    # By config
    print(f"\n{'Config':<20} {'Avg':>5} {'1':>4} {'2':>4} {'3':>4} {'4':>4} {'5':>4} {'N':>4}")
    print("-" * 60)
    by_config = defaultdict(list)
    for r in scored:
        by_config[r["config"]].append(r["score"])
    for config in sorted(by_config):
        scores = by_config[config]
        avg_s = sum(scores) / len(scores)
        dist = [scores.count(i) for i in range(1, 6)]
        print(f"{config:<20} {avg_s:>5.2f} {dist[0]:>4} {dist[1]:>4} {dist[2]:>4} {dist[3]:>4} {dist[4]:>4} {len(scores):>4}")

    # By config × category
    print(f"\n{'Config':<20} {'Category':<25} {'Avg':>5} {'N':>4}")
    print("-" * 60)
    by_cc = defaultdict(list)
    for r in scored:
        by_cc[(r["config"], r["category"])].append(r["score"])
    for (config, cat) in sorted(by_cc):
        scores = by_cc[(config, cat)]
        avg_s = sum(scores) / len(scores)
        print(f"{config:<20} {cat:<25} {avg_s:>5.2f} {len(scores):>4}")

    # By config × difficulty
    print(f"\n{'Config':<20} {'Difficulty':<15} {'Avg':>5} {'N':>4}")
    print("-" * 55)
    by_cd = defaultdict(list)
    for r in scored:
        by_cd[(r["config"], r["difficulty"])].append(r["score"])
    for (config, diff) in sorted(by_cd):
        scores = by_cd[(config, diff)]
        avg_s = sum(scores) / len(scores)
        print(f"{config:<20} {diff:<15} {avg_s:>5.2f} {len(scores):>4}")

    # By config × module
    print(f"\n{'Config':<20} {'Module':<15} {'Avg':>5} {'N':>4}")
    print("-" * 55)
    by_cm = defaultdict(list)
    for r in scored:
        by_cm[(r["config"], r["module"])].append(r["score"])
    for (config, mod) in sorted(by_cm):
        scores = by_cm[(config, mod)]
        avg_s = sum(scores) / len(scores)
        print(f"{config:<20} {mod:<15} {avg_s:>5.2f} {len(scores):>4}")


def main():
    parser = argparse.ArgumentParser(description="Score evaluation results with LLM judge")
    parser.add_argument("--results", required=True, nargs="+", help="Path(s) to eval result JSONL files")
    parser.add_argument("--judge", default="grok", choices=["grok", "anthropic", "openai"], help="LLM judge backend")
    parser.add_argument("--output", help="Output path for scored results (default: auto-named)")
    parser.add_argument("--limit", type=int, help="Limit number of results to score (for testing)")
    parser.add_argument("--judge-model", help="Override the judge model id (e.g. grok-4.3).")
    args = parser.parse_args()

    _load_env()
    judge_fn = {"grok": score_with_grok, "anthropic": score_with_anthropic, "openai": score_with_openai}[args.judge]
    if args.judge_model:
        base_fn = judge_fn
        judge_fn = lambda prompt: base_fn(prompt, args.judge_model)

    all_results = []
    for path in args.results:
        with open(path) as f:
            for line in f:
                all_results.append(json.loads(line))

    if args.limit:
        all_results = all_results[:args.limit]

    print(f"Scoring {len(all_results)} results with {args.judge} judge...")

    for i, result in enumerate(all_results, 1):
        if result.get("score", -1) > 0:
            continue
        score_result(result, judge_fn)
        print(f"  [{i}/{len(all_results)}] {result['task_id']} ({result['config']}): score={result.get('score', '?')}")
        time.sleep(0.1)

    # Save scored results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / f"scored_{Path(args.results[0]).stem}.jsonl"

    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nScored results → {output_path}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
