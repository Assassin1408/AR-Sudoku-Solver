"""Check that everything works on this machine, without needing a webcam.

    python check_setup.py            # libraries, model, solver, full pipeline, display
    python check_setup.py --camera   # also try to open the webcam
"""
import argparse
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

results = []


def check(name, fn, required=True):
    t = time.perf_counter()
    try:
        detail = fn() or ""
        ok = True
    except SystemExit as e:
        ok, detail = False, str(e)
    except Exception as e:  # report and keep going
        ok, detail = False, f"{type(e).__name__}: {e}"
    ms = 1000 * (time.perf_counter() - t)
    tag = "PASS" if ok else ("FAIL" if required else "WARN")
    print(f"[{tag}] {name:<26} {detail} ({ms:.0f} ms)", flush=True)
    results.append(ok or not required)
    return ok


def libs():
    import cv2
    import numpy
    import torch
    return (f"Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
            f"torch {torch.__version__}, OpenCV {cv2.__version__}, numpy {numpy.__version__}")


def model():
    from digit_model import DEFAULT_WEIGHTS, load_model, pick_device
    dev = pick_device()
    load_model(DEFAULT_WEIGHTS, dev)
    return f"weights loaded on {dev}"


def solver():
    from solver import solve
    p = ("8........" "..36....." ".7..9.2.." ".5...7..." "....457.."
         "...1...3." "..1....68" "..85...1." ".9....4..")
    b = [[int(ch) if ch != "." else 0 for ch in p[i * 9:i * 9 + 9]] for i in range(9)]
    sol, n = solve(b, count_limit=2)
    if sol is None or n != 1 or sol[0][1] != 1:
        raise RuntimeError("wrong solution")
    return "hardest known puzzle solved, unique"


def pipeline():
    import numpy as np
    from ar_sudoku import ARSudoku, solve_with_repair
    from digit_model import DEFAULT_WEIGHTS, load_model, pick_device
    from grid_detector import find_grid, read_board, rectify
    from synth import camera_shot, random_puzzle, render_board

    dev = pick_device()
    m = load_model(DEFAULT_WEIGHTS, dev)
    rng = np.random.default_rng(0)
    solved = 0
    n = 10
    for _ in range(n):
        puzzle, sol = random_puzzle(rng, clues=30)
        img, _ = camera_shot(render_board(puzzle, rng, font_kind="hershey"), rng)
        q = find_grid(img)
        if q is None:
            continue
        board, _, _ = rectify(img, q)
        digits, conf, probs = read_board(m, board, dev)
        s, _ = solve_with_repair(digits, conf, probs)
        solved += s is not None and (s == sol).all()
    # One frame through the live class to check the overlay path
    ARSudoku(m, dev).process(img)
    if solved < n // 2:
        raise RuntimeError(f"only {solved}/{n} synthetic photos solved")
    return f"{solved}/{n} synthetic camera photos solved from a single frame"


def display():
    import cv2
    import numpy as np
    from ar_sudoku import has_display
    if not has_display():
        raise RuntimeError("no display found (SSH or server); "
                           "run ar_sudoku.py with --no-show --save out.mp4")
    try:
        cv2.namedWindow("check", cv2.WINDOW_NORMAL)
        cv2.imshow("check", np.zeros((10, 10, 3), np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    except cv2.error:
        raise RuntimeError("no window support (headless machine or opencv-python-headless); "
                           "run ar_sudoku.py with --no-show --save out.mp4")
    return "windows can be opened"


def camera():
    from ar_sudoku import open_source
    cap, _ = open_source("0")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("camera opened but returned no frame")
    return f"camera 0 gives {frame.shape[1]}x{frame.shape[0]} frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="also test webcam 0")
    args = ap.parse_args()
    if not check("libraries", libs):
        raise SystemExit("Install the requirements first:  pip install -r requirements.txt")
    check("model weights", model)
    check("solver", solver)
    check("end to end pipeline", pipeline)
    check("display", display, required=False)
    if args.camera:
        check("camera", camera, required=False)
    print("\nSetup OK." if all(results) else "\nSome checks failed, see above.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
