"""End to end evaluation on synthetic camera shots.

    python tests/test_pipeline.py --n 200
    python tests/test_pipeline.py --n 50 --save-examples   # images in output/examples
    python tests/test_pipeline.py --n 200 --save           # also writes tests/results/accuracy.txt

Reports grid detection rate, cell level accuracy (empty vs digit and digit
class), fully correct board reads, and correct solutions (after repair)."""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from ar_sudoku import OUTPUT_DIR, draw_solution, solve_with_repair, write_image  # noqa: E402
from digit_model import DEFAULT_WEIGHTS, load_model  # noqa: E402
from grid_detector import find_grid, read_board, rectify  # noqa: E402
from synth import camera_shot, random_puzzle, render_board  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "tests", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--font", default=None, choices=[None, "hershey", "ttf"])
    ap.add_argument("--save-examples", nargs="?", const=os.path.join(OUTPUT_DIR, "examples"),
                    default=None, help="save the first 6 solved photos (default output/examples)")
    ap.add_argument("--save", nargs="?", const="accuracy.txt", default=None, metavar="NAME",
                    help="also save the results in tests/results/ (default accuracy.txt)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.save_examples and not os.path.isabs(args.save_examples):
        args.save_examples = os.path.join(OUTPUT_DIR, args.save_examples)
    model = load_model(args.weights)
    rng = np.random.default_rng(args.seed)
    found = cells = cell_ok = digit_cells = digit_ok = empty_ok = empty_cells = 0
    boards_ok = solved_ok = 0
    confusion = np.zeros((10, 10), int)
    for i in range(args.n):
        puzzle, sol = random_puzzle(rng, clues=int(rng.integers(24, 36)))
        img, _ = camera_shot(render_board(puzzle, rng, font_kind=args.font), rng)
        q = find_grid(img)
        if q is None:
            continue
        found += 1
        board, _, Minv = rectify(img, q)
        digits, conf, probs = read_board(model, board)
        cells += 81
        cell_ok += (digits == puzzle).sum()
        m = puzzle > 0
        digit_cells += m.sum()
        digit_ok += (digits[m] == puzzle[m]).sum()
        empty_cells += (~m).sum()
        empty_ok += (digits[~m] == 0).sum()
        np.add.at(confusion, (puzzle.ravel(), digits.ravel()), 1)
        boards_ok += (digits == puzzle).all()
        s, _ = solve_with_repair(digits, conf, probs)
        good = s is not None and (s == sol).all()
        solved_ok += good
        if args.save_examples and i < 6:
            os.makedirs(args.save_examples, exist_ok=True)
            out = draw_solution(img, Minv, puzzle, s) if s is not None else img
            write_image(os.path.join(args.save_examples, f"ex{i}_{'ok' if good else 'fail'}.jpg"), out)

    lines = [f"accuracy, {time.strftime('%Y-%m-%d %H:%M')}, {args.n} synthetic photos, "
             f"seed {args.seed}, weights {os.path.basename(args.weights)}",
             f"grids found        {found}/{args.n}",
             f"cell accuracy      {cell_ok / cells:.4f}",
             f"  digit cells      {digit_ok / digit_cells:.4f}  ({digit_cells} cells)",
             f"  empty cells      {empty_ok / empty_cells:.4f}  ({empty_cells} cells)",
             f"boards read 100%   {boards_ok}/{found}",
             f"correctly solved   {solved_ok}/{found}  (with low confidence repair)",
             "confusion (row = true, col = predicted, 0 = empty):",
             str(confusion)]
    print("\n".join(lines))
    if args.save:
        path = os.path.join(RESULTS_DIR, args.save)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("saved", path)


if __name__ == "__main__":
    main()
