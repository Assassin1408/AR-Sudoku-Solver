"""Simulate a handheld camera moving over a printed sudoku, run the live AR
pipeline frame by frame, and save the result.

    python tests/make_demo_video.py --out demo.mp4 --frames 120"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from ar_sudoku import ARSudoku, make_writer, write_image  # noqa: E402
from digit_model import DEFAULT_WEIGHTS, load_model  # noqa: E402
from synth import random_puzzle, render_board  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    puzzle, sol = random_puzzle(rng, clues=27)
    board = cv2.cvtColor(render_board(puzzle, rng, font_kind="ttf"), cv2.COLOR_GRAY2BGR)
    W, H = 960, 720
    bg = np.full((H, W, 3), (95, 120, 150), np.float32)
    bg += cv2.GaussianBlur(rng.normal(0, 30, (H, W)).astype(np.float32), (0, 0), 8)[..., None]
    bh, bw = board.shape[:2]
    src = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])

    app = ARSudoku(load_model(args.weights))
    writer, args.out = make_writer(args.out, 30, (W, H))
    times = []
    for f in range(args.frames):
        t = f / 30
        s = 430 + 50 * np.sin(0.7 * t)
        cx, cy = W / 2 + 120 * np.sin(0.9 * t), H / 2 + 50 * np.cos(0.6 * t)
        ang = 0.2 * np.sin(0.5 * t)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * s / 2
        tilt = np.array([[0, 0], [0, 40 * np.sin(1.1 * t)], [30 * np.cos(0.8 * t), 0],
                         [0, -30 * np.sin(0.7 * t)]])
        dst = (base @ R.T + tilt + [cx, cy]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        paper = cv2.warpPerspective(board, M, (W, H)).astype(np.float32)
        mask = cv2.warpPerspective(np.ones((bh, bw), np.float32), M, (W, H))[..., None]
        frame = bg * (1 - mask) + paper * mask
        frame += rng.normal(0, 4, frame.shape)
        frame = cv2.GaussianBlur(np.clip(frame, 0, 255).astype(np.uint8), (0, 0), 0.8)

        t0 = time.perf_counter()
        out = app.process(frame)
        times.append(time.perf_counter() - t0)
        cv2.putText(out, app.status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(out)
        if f == args.frames // 2:
            write_image(os.path.splitext(args.out)[0] + "_frame.jpg", out)
    writer.release()
    ms = 1000 * np.array(times)
    print(f"per frame: median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms "
          f"(~{1000 / np.median(ms):.0f} fps on this CPU); final status: {app.status}")


if __name__ == "__main__":
    main()
