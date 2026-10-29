"""Summarise metrics across repeated training runs (different seeds).

Usage: python summarize_runs.py [log_dir]   (default: logs_shallownet)

Each log ends with an "all result" table; we take its `mean` row
(accuracy, precision, recall, f1, kappa) as that run's score.
"""
import math
import sys
from pathlib import Path

log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "logs_shallownet")


def parse_run(path):
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "all result" in l)
    names = lines[start + 1].split()                     # metric names (header row)
    mean_row = next(l for l in lines[start:] if l.startswith("mean"))
    return dict(zip(names, map(float, mean_row.split()[1:])))


runs = sorted(log_dir.glob("run*.txt"))
results = [parse_run(p) for p in runs]
n = len(results)
print(f"{n} runs from {log_dir}\n")
print(f"{'metric':<10}{'mean':>9}{'std':>9}{'SE':>9}   per-run")

for metric in results[0]:
    x = [r[metric] for r in results]

    # mean:  x̄ = (1/n) Σ x_i
    mean = sum(x) / n

    # std:  s = sqrt( Σ (x_i - x̄)² / (n - 1) )
    # divide by n-1, not n: x̄ is estimated from the same runs, so dividing by n underestimates the spread
    std = math.sqrt(sum((xi - mean) ** 2 for xi in x) / (n - 1))

    # standard error of the mean:  SE = s / sqrt(n)
    # std = how much a single run varies; SE = how uncertain the mean x̄ is (shrinks as more runs are added)
    se = std / math.sqrt(n)

    print(f"{metric:<10}{mean:9.2f}{std:9.2f}{se:9.2f}   " + ", ".join(f"{xi:.2f}" for xi in x))
