"""Evaluation harness: run benchmark tasks through the full matrix.

Usage:
    python src/eval_harness.py --benchmark data/benchmark.jsonl --configs 3B-base,3B-RAG,API-ceiling
    python src/eval_harness.py --benchmark data/benchmark.jsonl --all
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from rag_pipeline import RAGPipeline, EVAL_CONFIGS

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "eval_results"


def load_benchmark(path: str) -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def run_config(config_name: str, tasks: list[dict]) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    pipeline = RAGPipeline.from_config(config_name)
    results = []

    for i, task in enumerate(tasks, 1):
        question = task["question"]
        print(f"  [{i}/{len(tasks)}] {task['id']}: {question[:80]}...")

        start = time.time()
        response = pipeline.query(question)
        elapsed = time.time() - start

        result = {
            "task_id": task["id"],
            "config": config_name,
            "model": response.model,
            "use_rag": pipeline.use_rag,
            "question": question,
            "expected_answer": task.get("expected_answer", ""),
            "generated_answer": response.answer,
            "category": task.get("category", ""),
            "module": task.get("module", ""),
            "difficulty": task.get("difficulty", ""),
            "num_chunks_retrieved": len(response.chunks),
            "chunk_distances": [c.distance for c in response.chunks],
            "latency_seconds": round(elapsed, 2),
        }
        results.append(result)
        print(f"         {elapsed:.1f}s | {len(response.answer)} chars")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--benchmark", help="Path to benchmark JSONL (default: data/benchmark_dev.jsonl)")
    parser.add_argument("--configs", help="Comma-separated config names (e.g. 3B-base,3B-RAG)")
    parser.add_argument("--all", action="store_true", help="Run all configs")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"],
                        help="Benchmark split (default: dev). Sets benchmark path if --benchmark not given.")
    parser.add_argument("--limit", type=int, help="Limit number of tasks (for quick testing)")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_path = args.benchmark
    else:
        split_map = {"dev": "benchmark_dev.jsonl", "test": "benchmark_test.jsonl", "full": "benchmark.jsonl"}
        benchmark_path = str(PROJECT_ROOT / "data" / split_map[args.split])

    tasks = load_benchmark(benchmark_path)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} benchmark tasks from {Path(benchmark_path).name}")

    if args.all:
        config_names = list(EVAL_CONFIGS.keys())
    elif args.configs:
        config_names = [c.strip() for c in args.configs.split(",")]
    else:
        parser.error("Specify --configs or --all")

    # Validate configs
    for name in config_names:
        if name not in EVAL_CONFIGS:
            parser.error(f"Unknown config: {name}. Available: {list(EVAL_CONFIGS.keys())}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    output_path = RESULTS_DIR / f"eval_{args.split}_{timestamp}.jsonl"

    # Append after every config rather than writing once at the end: a full matrix
    # run is ~1 h of GPU time, and a process kill partway through used to discard
    # every completed config with it.
    for config_name in config_names:
        try:
            results = run_config(config_name, tasks)
            all_results.extend(results)
            with open(output_path, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"  saved {len(results)} results for {config_name} → {output_path.name}")
        except Exception as e:
            print(f"\n  ERROR running {config_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Results: {len(all_results)} total → {output_path}")

    # Summary table
    print(f"\n{'Config':<20} {'Tasks':>6} {'Avg Latency':>12}")
    print("-" * 40)
    for config_name in config_names:
        config_results = [r for r in all_results if r["config"] == config_name]
        if config_results:
            avg_lat = sum(r["latency_seconds"] for r in config_results) / len(config_results)
            print(f"{config_name:<20} {len(config_results):>6} {avg_lat:>10.1f}s")


if __name__ == "__main__":
    main()
