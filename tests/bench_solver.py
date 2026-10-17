"""Speed check for the sudoku solver (no camera or model needed).

    python tests/bench_solver.py
    python tests/bench_solver.py --repeat 50

Times solve() on easy, medium and hard puzzles, including the uniqueness
check (count_limit=2) that the app can use."""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import solve  # noqa: E402

PUZZLES = {
    "easy": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "medium": "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
    "hard": "400000805030000000000700000020000060000080400000010000000603070500200000104000000",
    "Inkala (world's hardest)": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
}


def grid(s):
    return np.array([int(ch) for ch in s]).reshape(9, 9)


def time_it(board, limit, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        sol, n = solve(board, count_limit=limit)
        times.append((time.perf_counter() - t) * 1000)
    return sol, n, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()
    print(f"{'puzzle':26} {'clues':>5} {'solved':>6} {'median ms':>10} {'worst ms':>9} {'unique check ms':>16}")
    for name, s in PUZZLES.items():
        b = grid(s)
        sol, _, t1 = time_it(b, 1, args.repeat)
        _, _, t2 = time_it(b, 2, args.repeat)
        print(f"{name:26} {int((b > 0).sum()):5d} {str(sol is not None):>6} "
              f"{statistics.median(t1):10.2f} {max(t1):9.2f} {statistics.median(t2):16.2f}")


if __name__ == "__main__":
    main()
