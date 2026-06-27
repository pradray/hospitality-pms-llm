"""Split benchmark.jsonl into dev (70%) and test (30%) sets.

Stratified by module × category to ensure proportional representation.
Deterministic split via fixed random seed.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "benchmark_dev.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "benchmark_test.jsonl"

SEED = 42
TEST_RATIO = 0.30


def main():
    with open(BENCHMARK_PATH) as f:
        tasks = [json.loads(line) for line in f]

    # Group by module × category for stratified split
    groups = defaultdict(list)
    for t in tasks:
        key = (t["module"], t["category"])
        groups[key].append(t)

    rng = random.Random(SEED)
    dev_tasks = []
    test_tasks = []

    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * TEST_RATIO))
        test_tasks.extend(group[:n_test])
        dev_tasks.extend(group[n_test:])

    # Write splits
    for path, split_tasks, label in [
        (DEV_PATH, dev_tasks, "dev"),
        (TEST_PATH, test_tasks, "test"),
    ]:
        with open(path, "w") as f:
            for t in split_tasks:
                f.write(json.dumps(t) + "\n")

    print(f"Total: {len(tasks)} tasks")
    print(f"Dev:   {len(dev_tasks)} tasks → {DEV_PATH.name}")
    print(f"Test:  {len(test_tasks)} tasks → {TEST_PATH.name}")

    # Show distribution
    print("\nDev split distribution:")
    _print_stats(dev_tasks)
    print("\nTest split distribution:")
    _print_stats(test_tasks)


def _print_stats(tasks):
    by_cat = defaultdict(int)
    by_mod = defaultdict(int)
    by_diff = defaultdict(int)
    for t in tasks:
        by_cat[t["category"]] += 1
        by_mod[t["module"]] += 1
        by_diff[t["difficulty"]] += 1

    print(f"  By category: {dict(sorted(by_cat.items()))}")
    print(f"  By module:   {dict(sorted(by_mod.items()))}")
    print(f"  By difficulty:{dict(sorted(by_diff.items()))}")


if __name__ == "__main__":
    main()
