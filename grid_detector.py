"""Find the sudoku grid in a camera frame, rectify it, and read the 81 cells."""
import cv2
import numpy as np

from digit_model import extract_digit, predict_digits

CELL = 56          # pixels per cell in the rectified board
SIDE = CELL * 9    # 504


def order_corners(pts):
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()],
                     pts[s.argmax()], pts[d.argmax()]], np.float32)


def _quad_ok(q, frame_area, min_area_frac):
    if not cv2.isContourConvex(q.astype(np.int32).reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(q)
    if area < min_area_frac * frame_area:
        return False
    sides = np.linalg.norm(q - np.roll(q, -1, 0), axis=1)
    # Grid should look roughly square even under perspective
    return sides.min() / sides.max() > 0.5


def find_grid(frame, min_area_frac=0.04):
    """Return the 4 ordered corners of the largest square-ish quadrilateral
    (the sudoku border), or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape
    scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    # RETR_LIST keeps nested contours, so both the paper edge and the grid's
    # thick outer border show up; the grid is the smallest large quad.
    # [-2] works with both the OpenCV 3 (3 values) and OpenCV 4 (2 values) API
    contours = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    area = small.shape[0] * small.shape[1]
    quads = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        a = cv2.contourArea(cnt)
        if a < min_area_frac * area:
            break
        hull = cv2.convexHull(cnt)
        # A grid border hugs its hull; a cell's hole or a blob does not
        if a < 0.85 * cv2.contourArea(hull):
            continue
        peri = cv2.arcLength(hull, True)
        for eps in (0.02, 0.03, 0.05, 0.08):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                q = order_corners(approx)
                if _quad_ok(q, area, min_area_frac):
                    quads.append((cv2.contourArea(q), q))
                break
    if not quads:
        return None
    biggest = max(a for a, _ in quads)
    candidates = [(a, q) for a, q in quads if a > 0.55 * biggest]
    for _, q in sorted(candidates, key=lambda t: t[0]):
        if _looks_like_grid(small, q):
            return q / scale
    return None


def _looks_like_grid(gray, q, n=180):
    """Rectify the candidate and check for dark lines at the 1/3 and 2/3
    positions (the box borders), which a plain sheet of paper lacks."""
    M = cv2.getPerspectiveTransform(np.float32(q),
                                    np.float32([[0, 0], [n, 0], [n, n], [0, n]]))
    w = cv2.warpPerspective(gray, M, (n, n)).astype(np.float32)
    rows, cols = w.mean(1), w.mean(0)
    ref = np.median(w)
    hits = 0
    for prof in (rows, cols):
        for k in (1, 2):
            c = int(k * n / 3)
            if prof[c - 4:c + 5].min() < ref - 15:
                hits += 1
    return hits >= 3


def refine_corners(gray, q):
    """Sub pixel refinement of the 4 outer corners."""
    q = q.reshape(-1, 1, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05)
    try:
        cv2.cornerSubPix(gray, q, (5, 5), (-1, -1), crit)
    except cv2.error:
        pass
    return q.reshape(4, 2)


def warp_matrices(corners):
    dst = np.float32([[0, 0], [SIDE - 1, 0], [SIDE - 1, SIDE - 1], [0, SIDE - 1]])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    return M, np.linalg.inv(M)


def rectify(frame, corners):
    M, Minv = warp_matrices(corners)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.warpPerspective(gray, M, (SIDE, SIDE)), M, Minv


def read_board(model, board_img, device="cpu"):
    """board_img: SIDE x SIDE rectified grayscale grid.
    Returns digits (9x9, 0 = empty), confidence (9x9) and probs (9x9x10)."""
    digits = np.zeros((9, 9), int)
    conf = np.ones((9, 9))
    probs = np.zeros((9, 9, 10))
    imgs, pos = [], []
    for r in range(9):
        for c in range(9):
            cell = board_img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
            d = extract_digit(cell)
            if d is not None:
                imgs.append(d)
                pos.append((r, c))
    top, p, full = predict_digits(model, imgs, device)
    for (r, c), d, pr, pv in zip(pos, top, p, full):
        digits[r, c], conf[r, c], probs[r, c] = d, pr, pv
    return digits, conf, probs
