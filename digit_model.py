"""MNIST digit classifier and the cell preprocessing that turns a sudoku cell
into an MNIST style 28x28 image (white digit on black, 20x20 box, centred by
centre of mass)."""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "mnist_cnn.pt")


class DigitCNN(nn.Module):
    """Small VGG style CNN, ~99.5% on the MNIST test set."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def pick_device(pref="auto"):
    """'auto' uses an NVIDIA GPU when present, otherwise the CPU. The model is
    small enough that the CPU keeps up with a webcam; pass 'mps' to try the
    Apple Silicon GPU."""
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path=DEFAULT_WEIGHTS, device="cpu"):
    if not os.path.exists(path):
        raise SystemExit(
            f"Model weights not found at {path}\n"
            "Train them with:  python download_mnist.py  then  python train_mnist.py --printed 20000")
    model = DigitCNN()
    # Load on CPU first so weights saved on any device load on any machine
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    return model


def to_mnist_format(digit_mask):
    """digit_mask: binary uint8 image (digit=255) cropped to the digit's
    bounding box. Returns a 28x28 float image in MNIST style."""
    h, w = digit_mask.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    digit = cv2.resize(digit_mask, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = digit
    # MNIST digits are centred by centre of mass, not bounding box
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        shift = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, shift, (28, 28))
    return canvas / 255.0


def extract_digit(cell, min_area_frac=0.025):
    """cell: grayscale crop of one sudoku cell (any size, roughly square).
    Returns a 28x28 MNIST style image, or None if the cell is empty."""
    h, w = cell.shape
    blur = cv2.GaussianBlur(cell, (3, 3), 0)
    block = max(3, (min(h, w) // 2) | 1)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 7)
    # Remove grid line remnants along the borders
    m = max(2, int(0.08 * min(h, w)))
    th[:m, :] = 0
    th[-m:, :] = 0
    th[:, :m] = 0
    th[:, -m:] = 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_frac * h * w or bh < 0.25 * h:
            continue
        cx, cy = cents[i]
        # Digit should sit near the centre of the cell
        if abs(cx - w / 2) > 0.3 * w or abs(cy - h / 2) > 0.3 * h:
            continue
        score = area / (1 + np.hypot(cx - w / 2, cy - h / 2))
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None

    x, y, bw, bh, _ = stats[best]
    # Keep pieces that belong to the same glyph (e.g. a broken stroke)
    keep = np.zeros_like(th)
    for i in range(1, n):
        xi, yi, wi, hi, ai = stats[i]
        if ai < 0.005 * h * w:
            continue
        if xi >= x - 0.1 * w and xi + wi <= x + bw + 0.1 * w and \
           yi >= y - 0.1 * h and yi + hi <= y + bh + 0.1 * h:
            keep[labels == i] = 255
    ys, xs = np.nonzero(keep)
    crop = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # Thicken thin printed strokes towards MNIST pen width
    stroke = max(1, int(round(min(h, w) / 28)))
    crop = cv2.copyMakeBorder(crop, stroke, stroke, stroke, stroke, cv2.BORDER_CONSTANT)
    crop = cv2.dilate(crop, np.ones((stroke, stroke), np.uint8)) if stroke > 1 else crop
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    return to_mnist_format(crop)


@torch.no_grad()
def predict_digits(model, images, device="cpu"):
    """images: list of 28x28 float arrays in [0,1].
    Returns (top1 digits, top1 probs, full prob matrix)."""
    if not images:
        return np.array([], int), np.array([]), np.zeros((0, 10))
    x = torch.from_numpy(np.stack(images)).float().unsqueeze(1)
    x = (x - MNIST_MEAN) / MNIST_STD
    probs = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
    # 0 never appears in sudoku, so renormalise over 1..9
    probs[:, 0] = 0
    probs /= probs.sum(1, keepdims=True)
    return probs.argmax(1), probs.max(1), probs
