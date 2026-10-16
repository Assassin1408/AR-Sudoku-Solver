"""Synthetic 'camera photos' of sudoku puzzles for testing the pipeline."""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import solve  # noqa: E402

_LINUX_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _test_fonts():
    """Fonts for TTF boards; falls back to system fonts on macOS/Windows.
    The Hershey fonts below are never seen in training."""
    fonts = [f for f in _LINUX_FONTS if os.path.exists(f)]
    if not fonts:
        from train_mnist import find_fonts
        fonts = find_fonts()[:8]
    return fonts


TEST_FONTS = _test_fonts()
HERSHEY = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
           cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]


def random_puzzle(rng, clues=28):
    """Random full grid (seeded backtracking on a shuffled first row), then
    blank cells down to the requested clue count."""
    b = np.zeros((9, 9), int)
    b[0] = rng.permutation(9) + 1
    sol, _ = solve(b)
    # Shuffle rows within bands and the bands themselves (keeps validity)
    rows = np.concatenate([rng.permutation(3) + 3 * band for band in rng.permutation(3)])
    sol = sol[rows]
    # Relabel digits randomly so solutions vary
    perm = rng.permutation(9) + 1
    sol = perm[sol - 1]
    if rng.random() < 0.5:
        sol = sol.T
    # Remove clues one at a time, keeping only removals that leave a unique
    # solution, so the ground truth answer is the only correct answer
    puzzle = sol.copy()
    for i in rng.permutation(81):
        if (puzzle > 0).sum() <= clues:
            break
        keep = puzzle.flat[i]
        puzzle.flat[i] = 0
        if solve(puzzle, count_limit=2)[1] != 1:
            puzzle.flat[i] = keep
    return puzzle, sol


def render_board(puzzle, rng, side=540, font_kind=None):
    cell = side // 9
    paper = int(rng.integers(215, 256))
    img = np.full((side + 20, side + 20), paper, np.uint8)
    o = 10
    for i in range(10):
        t = 4 if i % 3 == 0 else 1
        p = o + i * cell
        cv2.line(img, (o, p), (o + 9 * cell, p), 20, t)
        cv2.line(img, (p, o), (p, o + 9 * cell), 20, t)
    kind = font_kind or ("hershey" if rng.random() < 0.5 else "ttf")
    if kind == "ttf" and not TEST_FONTS:
        kind = "hershey"  # no usable system fonts on this machine
    if kind == "hershey":
        f = HERSHEY[rng.integers(len(HERSHEY))]
        for r in range(9):
            for c in range(9):
                d = puzzle[r, c]
                if d:
                    s, th = cell / 36, max(2, cell // 22)
                    (tw, tht), _ = cv2.getTextSize(str(d), f, s, th)
                    cv2.putText(img, str(d), (o + c * cell + (cell - tw) // 2,
                                              o + r * cell + (cell + tht) // 2), f, s, 15, th,
                                cv2.LINE_AA)
    else:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(TEST_FONTS[rng.integers(len(TEST_FONTS))], int(cell * 0.65))
        for r in range(9):
            for c in range(9):
                d = puzzle[r, c]
                if d:
                    x0, y0, x1, y1 = draw.textbbox((0, 0), str(d), font=font)
                    draw.text((o + c * cell + (cell - (x1 - x0)) / 2 - x0,
                               o + r * cell + (cell - (y1 - y0)) / 2 - y0), str(d),
                              fill=15, font=font)
        img = np.array(pil)
    return img


def camera_shot(board, rng, out_wh=(960, 720), tilt=0.12, scale=(0.55, 0.85)):
    """Place the board in a cluttered scene with perspective, lighting
    gradient, blur, sensor noise and JPEG compression. Returns BGR image and
    the true outer grid corners."""
    W, H = out_wh
    bg = rng.integers(60, 200, 3).astype(np.float32)
    scene = np.ones((H, W, 3), np.float32) * bg
    scene += cv2.GaussianBlur(rng.normal(0, 25, (H, W)).astype(np.float32),
                              (0, 0), 6)[..., None]
    for _ in range(6):  # clutter
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(scene, (int(x), int(y)), int(rng.integers(10, 60)),
                   rng.integers(0, 255, 3).tolist(), -1)

    while True:  # resample until the whole sheet is inside the frame
        s = min(W, H) * rng.uniform(*scale)
        cx, cy = W / 2 + rng.uniform(-0.12, 0.12) * W, H / 2 + rng.uniform(-0.08, 0.08) * H
        base = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * s / 2
        ang = rng.uniform(-0.25, 0.25)
        R = np.float32([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        dst = base @ R.T + rng.uniform(-tilt, tilt, (4, 2)).astype(np.float32) * s + [cx, cy]
        if dst.min() > 5 and (dst[:, 0] < W - 5).all() and (dst[:, 1] < H - 5).all():
            break
    bh, bw = board.shape
    src = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    paper = cv2.warpPerspective(cv2.cvtColor(board, cv2.COLOR_GRAY2BGR).astype(np.float32),
                                M, (W, H))
    mask = cv2.warpPerspective(np.ones((bh, bw), np.float32), M, (W, H))[..., None]
    img = scene * (1 - mask) + paper * mask
    # Lighting gradient
    gx, gy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    light = 0.7 + 0.4 * (rng.uniform(-1, 1) * gx + rng.uniform(-1, 1) * gy) / 2 + 0.15
    img *= light[..., None]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.4))
    img += rng.normal(0, rng.uniform(2, 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 92))])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # Outer grid corners: board has a 10px margin
    corners_src = np.float32([[10, 10], [bw - 11, 10], [bw - 11, bh - 11], [10, bh - 11]])
    corners = cv2.perspectiveTransform(corners_src[None], M)[0]
    return img, corners
