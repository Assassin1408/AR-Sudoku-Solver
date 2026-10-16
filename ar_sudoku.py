"""Live AR sudoku solver.

    python ar_sudoku.py                    # webcam 0
    python ar_sudoku.py --source 1         # another camera
    python ar_sudoku.py --source clip.mp4 --save           # output/solved_<time>.mp4
    python ar_sudoku.py --source photo.jpg --save solved.jpg  # output/solved.jpg
    python ar_sudoku.py --no-show --save               # machines without a display

Everything is saved inside the project's output/ folder (git ignored):
results from --save, and snapshots in output/snapshots/. An absolute --save
path is used as given.

Keys: q or Esc quit, r reset the current reading, s save a snapshot.
Works on Windows, macOS and Linux."""
import argparse
import os
import platform
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digit_model import DEFAULT_WEIGHTS, ROOT, load_model, pick_device  # noqa: E402
from grid_detector import CELL, SIDE, find_grid, rectify, read_board  # noqa: E402
from solver import is_valid_givens, solve  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
OUTPUT_DIR = os.path.join(ROOT, "output")


def output_path(name=None, default_ext=".png", prefix="solved", subdir=""):
    """Resolve where a result is saved. Relative names go inside output/
    (or output/<subdir>/), absolute paths are kept, and no name gives a
    timestamped one that never overwrites an earlier file. The parent folder
    is created when missing."""
    auto = not name
    if auto:
        name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}{default_ext}"
    elif not os.path.splitext(name)[1]:
        name += default_ext
    path = name if os.path.isabs(name) else os.path.join(OUTPUT_DIR, subdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if auto:
        stem, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(path):
            path = f"{stem}_{i}{ext}"
            i += 1
    return path


# ---------------------------------------------------------------- file I/O
# cv2.imread/imwrite fail silently on Windows paths with non ASCII characters,
# so images go through numpy, which handles any path on any OS.
def read_image(path):
    data = np.fromfile(path, dtype=np.uint8) if os.path.exists(path) else None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data is not None and data.size else None
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def write_image(path, img):
    if not os.path.splitext(path)[1]:
        path += ".png"
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise SystemExit(f"Could not encode image: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buf.tofile(path)
    return path


# ------------------------------------------------------------------ camera
def camera_backends():
    """Capture backends to try, in order, for this OS."""
    system = platform.system()
    if system == "Windows":
        # DirectShow opens most webcams instantly; MSMF can take ~10 s or fail
        names = ["CAP_DSHOW", "CAP_MSMF"]
    elif system == "Darwin":
        names = ["CAP_AVFOUNDATION"]
    else:
        names = ["CAP_V4L2"]
    return [getattr(cv2, n) for n in names if hasattr(cv2, n)] + [cv2.CAP_ANY]


def quiet_opencv_logs():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:  # older OpenCV builds without the logging module
        pass


def open_source(source):
    """Open a camera index, video file or stream URL. Returns (capture, is_camera)."""
    if source.isdigit():
        quiet_opencv_logs()  # probing backends prints scary but harmless warnings
        for backend in camera_backends():
            cap = cv2.VideoCapture(int(source), backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                for _ in range(20):  # some cameras return a few empty frames first
                    ok, _ = cap.read()
                    if ok:
                        return cap, True
                    time.sleep(0.05)
            cap.release()
        hint = {
            "Darwin": "allow camera access for your terminal or IDE in System Settings > "
                      "Privacy & Security > Camera, then restart it",
            "Windows": "check Settings > Privacy & security > Camera, and close other apps "
                       "using the camera (Teams, Zoom)",
        }.get(platform.system(), "check that your user can access /dev/video* "
                                 "(e.g. add it to the 'video' group)")
        raise SystemExit(f"Cannot open camera {source}. Try --source 1, or {hint}.")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    return cap, False


# ----------------------------------------------------------------- display
def has_display():
    """On Linux, OpenCV's Qt/GTK window code aborts the whole process when
    there is no X11 or Wayland display, so check before opening a window."""
    if platform.system() == "Linux":
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class Display:
    """cv2.imshow wrapper that falls back to no window when the machine has no
    display (SSH, Docker, Colab, opencv-python-headless) instead of crashing."""

    def __init__(self, enabled, title="AR Sudoku"):
        self.enabled, self.title = enabled, title
        if enabled and not has_display():
            self.enabled = False
            print("No display found, continuing without a window. "
                  "Use --save to record it in output/.", flush=True)

    def show(self, img):
        if not self.enabled:
            return
        try:
            cv2.imshow(self.title, img)
        except cv2.error:
            self.enabled = False
            print("No display available, continuing without a window. "
                  "Use --save to record it in output/.", flush=True)

    def key(self, wait_ms=1):
        if not self.enabled:
            return -1
        k = cv2.waitKey(wait_ms) & 0xFF
        # Closing the window with its X button counts as quit
        try:
            if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
                return ord("q")
        except cv2.error:
            pass
        return k

    def close(self):
        if self.enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def solve_with_repair(digits, conf, probs, max_tries=8):
    """Solve the read board. If misreads make it invalid or unsolvable, swap
    the least confident givens for their second best class, one at a time."""
    sol, _ = solve(digits)
    if sol is not None:
        return sol, digits
    givens = sorted(zip(*np.nonzero(digits)), key=lambda rc: conf[rc])
    for r, c in givens[:max_tries]:
        alt = np.argsort(probs[r, c])[::-1]
        for d in alt[1:3]:
            if d == 0:
                continue
            trial = digits.copy()
            trial[r, c] = d
            if is_valid_givens(trial):
                sol, _ = solve(trial)
                if sol is not None:
                    return sol, trial
    return None, digits


def draw_solution(frame, Minv, givens, solution, color=(40, 200, 40)):
    """Render the missing digits in rectified board space, then warp them back
    onto the camera frame with the inverse homography."""
    canvas = np.zeros((SIDE, SIDE, 3), np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, CELL / 40, max(2, CELL // 18)
    for r in range(9):
        for c in range(9):
            if givens[r, c] == 0:
                txt = str(int(solution[r, c]))
                (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
                org = (c * CELL + (CELL - tw) // 2, r * CELL + (CELL + th) // 2)
                cv2.putText(canvas, txt, org, font, scale, color, thick, cv2.LINE_AA)
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(canvas, Minv, (w, h))
    mask = warped.any(axis=2)
    out = frame.copy()
    out[mask] = warped[mask]
    return out


class ARSudoku:
    """Per frame: detect grid, read digits, vote across frames, solve once per
    distinct board, and overlay the cached solution."""

    def __init__(self, model, device="cpu", vote_frames=6, lost_reset=15):
        self.model, self.device = model, device
        self.votes = deque(maxlen=vote_frames)
        self.lost, self.lost_reset = 0, lost_reset
        self.corners = None
        self.cache = {}
        self.status = "searching"

    def reset(self):
        self.votes.clear()
        self.corners = None

    def _smooth(self, q):
        if self.corners is not None and np.abs(q - self.corners).mean() < 6:
            q = 0.6 * self.corners + 0.4 * q   # damp jitter when nearly still
        self.corners = q
        return q

    def process(self, frame):
        q = find_grid(frame)
        if q is None:
            self.lost += 1
            if self.lost > self.lost_reset:
                self.reset()
            self.status = "searching for grid"
            return frame
        self.lost = 0
        q = self._smooth(q)
        board, M, Minv = rectify(frame, q)
        digits, conf, probs = read_board(self.model, board, self.device)
        self.votes.append((digits, conf, probs))

        # Per cell majority vote over recent frames
        stack = np.stack([v[0] for v in self.votes])
        voted = np.zeros((9, 9), int)
        for r in range(9):
            for c in range(9):
                voted[r, c] = np.bincount(stack[:, r, c], minlength=10).argmax()
        mconf = np.mean([v[1] for v in self.votes], axis=0)
        mprobs = np.mean([v[2] for v in self.votes], axis=0)

        key = voted.tobytes()
        if key not in self.cache:
            self.cache[key] = solve_with_repair(voted, mconf, mprobs)
        sol, used = self.cache[key]

        out = frame.copy()
        cv2.polylines(out, [q.astype(np.int32)], True, (255, 160, 0), 2)
        if sol is None:
            self.status = f"read {int((voted > 0).sum())} clues, no valid solution yet"
            return out
        self.status = f"solved ({int((used > 0).sum())} clues)"
        return draw_solution(out, Minv, used, sol)


def make_writer(path, fps, size):
    """mp4v is available in almost every OpenCV build; if not, fall back to
    MJPG in an .avi container, which every build can write."""
    fps = fps if fps and 1 <= fps <= 240 else 30
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if w.isOpened():
        return w, path
    alt = os.path.splitext(path)[0] + ".avi"
    w = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not w.isOpened():
        raise SystemExit(f"Could not open a video writer for {path}")
    print(f"mp4 encoding unavailable here, saving to {alt} instead", flush=True)
    return w, alt


def main():
    ap = argparse.ArgumentParser(description="Live AR sudoku solver")
    ap.add_argument("--source", default="0",
                    help="camera index (0, 1, ...), video file, stream URL or image path")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--save", nargs="?", const="", default=None, metavar="NAME",
                    help="save the result in output/ (a name is optional; "
                         "without one a timestamped name is used)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    args = ap.parse_args()

    device = pick_device(args.device)
    model = load_model(args.weights, device)
    app = ARSudoku(model, device)
    display = Display(not args.no_show)

    is_image = os.path.splitext(args.source)[1].lower() in IMAGE_EXTS
    if args.save is not None:
        args.save = output_path(args.save, ".jpg" if is_image else ".mp4")

    if is_image:
        out = app.process(read_image(args.source))
        print(app.status)
        if args.save:
            print("saved", write_image(args.save, out))
        display.show(out)
        if display.enabled:
            while display.key(50) not in (ord("q"), 27):  # q or Esc
                pass
        display.close()
        return

    cap, is_camera = open_source(args.source)
    print(f"Running on {device}. Press q or Esc in the window to quit."
          if display.enabled else "Running without a window. Press Ctrl+C to stop.",
          flush=True)
    writer, t_prev, fps, fails, last_status = None, time.time(), 0.0, 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if not is_camera:
                    break  # end of the video file
                fails += 1  # cameras drop frames now and then
                if fails > 60:
                    print("Camera stopped sending frames.", flush=True)
                    break
                time.sleep(0.02)
                continue
            fails = 0
            out = app.process(frame)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            cv2.putText(out, f"{app.status} | {fps:.0f} fps", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if app.status != last_status:
                print(app.status, flush=True)
                last_status = app.status
            if args.save:
                if writer is None:
                    h, w = out.shape[:2]
                    writer, args.save = make_writer(args.save, cap.get(cv2.CAP_PROP_FPS), (w, h))
                writer.write(out)
            display.show(out)
            k = display.key(1)
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                app.reset()
            if k == ord("s"):
                path = output_path(prefix="snapshot", subdir="snapshots")
                print("saved", write_image(path, out), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
            print("saved", args.save)
        display.close()


if __name__ == "__main__":
    main()
