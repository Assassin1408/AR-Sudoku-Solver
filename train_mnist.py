"""Train the digit CNN on MNIST.

    python train_mnist.py                      # MNIST only
    python train_mnist.py --printed 20000      # MNIST + synthetic printed digits

MNIST is read straight from the raw idx files in data/MNIST/raw (downloaded
by download_mnist.py), so torchvision is not required.

The --printed option mixes in printed digits rendered from system fonts and
passed through the exact same cell preprocessing used at inference. MNIST is
handwritten, while most sudoku grids are printed, so this closes the domain
gap (see README for the measured effect)."""
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from digit_model import DEFAULT_WEIGHTS, ROOT, DigitCNN, MNIST_MEAN, MNIST_STD

DATA_DIR = os.path.join(ROOT, "data", "MNIST", "raw")


def load_idx(path, offset):
    return np.fromfile(path, dtype=np.uint8, offset=offset)


def load_mnist(root=DATA_DIR):
    if not os.path.exists(os.path.join(root, "train-images-idx3-ubyte")):
        raise SystemExit(f"MNIST not found in {root}. Run:  python download_mnist.py")
    p = lambda name: os.path.join(root, name)  # noqa: E731
    xtr = load_idx(p("train-images-idx3-ubyte"), 16).reshape(-1, 28, 28)
    ytr = load_idx(p("train-labels-idx1-ubyte"), 8)
    xte = load_idx(p("t10k-images-idx3-ubyte"), 16).reshape(-1, 28, 28)
    yte = load_idx(p("t10k-labels-idx1-ubyte"), 8)
    return (xtr.astype(np.float32) / 255, ytr.astype(np.int64),
            xte.astype(np.float32) / 255, yte.astype(np.int64))


def font_dirs():
    """System and per user font folders on Windows, macOS and Linux."""
    home = os.path.expanduser("~")
    windir = os.environ.get("WINDIR", r"C:\Windows")
    local = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
    return [
        os.path.join(windir, "Fonts"),                              # Windows
        os.path.join(local, "Microsoft", "Windows", "Fonts"),       # Windows (user)
        "/System/Library/Fonts", "/Library/Fonts",                  # macOS
        os.path.join(home, "Library", "Fonts"),                     # macOS (user)
        "/usr/share/fonts", "/usr/local/share/fonts",               # Linux
        os.path.join(home, ".fonts"),
        os.path.join(home, ".local", "share", "fonts"),
    ]


def find_fonts():
    fonts = set()
    for d in font_dirs():
        if os.path.isdir(d):
            for ext in ("ttf", "otf", "ttc", "TTF", "OTF", "TTC"):
                fonts.update(glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True))
    # Skip symbol, CJK and icon fonts that have no usable Latin digits
    bad = ("symbol", "emoji", "dingbat", "cjk", "math", "icon", "awesome", "wingding",
           "webdings", "marlett", "lastresort", "braille")
    fonts = [f for f in sorted(fonts) if not any(b in os.path.basename(f).lower() for b in bad)]
    return [f for f in fonts if _has_distinct_digits(f)]


def _has_distinct_digits(path):
    """Reject fonts whose digits are missing or render as identical
    placeholder boxes."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(path, 32)
        renders = set()
        for d in range(1, 10):
            img = Image.new("L", (48, 48), 0)
            ImageDraw.Draw(img).text((4, 2), str(d), fill=255, font=font)
            renders.add(img.tobytes())
    except Exception:  # unreadable or unsupported font file
        return False
    return len(renders) == 9


def make_printed_digits(n, seed=0):
    """Render digits 1..9 in random fonts inside a grid cell, add noise and
    blur, then run them through extract_digit so they look exactly like
    what the live pipeline feeds the model."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from digit_model import extract_digit

    rng = np.random.default_rng(seed)
    fonts = find_fonts()
    if not fonts:
        raise RuntimeError("No .ttf/.otf fonts found for printed digits")
    xs, ys = [], []
    while len(xs) < n:
        d = int(rng.integers(1, 10))
        size = int(rng.integers(40, 80))
        img = Image.new("L", (size, size), int(rng.integers(170, 256)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(fonts[rng.integers(len(fonts))],
                                      int(size * rng.uniform(0.5, 0.8)))
        except OSError:
            continue
        x0, y0, x1, y1 = draw.textbbox((0, 0), str(d), font=font)
        if x1 - x0 < 2 or y1 - y0 < 0.3 * size:
            continue
        ox = (size - (x1 - x0)) / 2 - x0 + rng.normal(0, size * 0.04)
        oy = (size - (y1 - y0)) / 2 - y0 + rng.normal(0, size * 0.04)
        draw.text((ox, oy), str(d), fill=int(rng.integers(0, 80)), font=font)
        a = np.array(img)
        # Partial grid lines on the cell border
        if rng.random() < 0.7:
            t = int(rng.integers(1, 4))
            a[:t, :] = 0
            a[:, :t] = 0
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.3))
        a = np.clip(a + rng.normal(0, rng.uniform(0, 12), a.shape), 0, 255).astype(np.uint8)
        m = extract_digit(a)
        if m is None:
            continue
        xs.append(m.astype(np.float32))
        ys.append(d)
    return np.stack(xs), np.array(ys, np.int64)


def augment(x, rng):
    """Random affine (rotation, scale, shear, shift) plus random stroke
    thickness change, applied on the batch with affine_grid."""
    b = x.shape[0]
    ang = torch.from_numpy(rng.uniform(-12, 12, b)).float() * math.pi / 180
    sc = torch.from_numpy(rng.uniform(0.85, 1.15, b)).float()
    sh = torch.from_numpy(rng.uniform(-0.15, 0.15, b)).float()
    tx = torch.from_numpy(rng.uniform(-0.12, 0.12, (b, 2))).float()
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([
        torch.stack([cos, -sin + sh, tx[:, 0]], 1),
        torch.stack([sin, cos, tx[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    r = rng.random(b)
    thick, thin = torch.from_numpy(r < 0.25), torch.from_numpy(r > 0.85)
    if thick.any():
        x[thick] = F.max_pool2d(x[thick], 3, 1, 1)
    if thin.any():
        x[thin] = -F.max_pool2d(-x[thin], 2, 1, 1)[..., :28, :28]
    return x


def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = (torch.from_numpy(x[i:i + bs]).unsqueeze(1) - MNIST_MEAN) / MNIST_STD
            correct += (model(xb).argmax(1).numpy() == y[i:i + bs]).sum()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--printed", type=int, default=0,
                    help="number of synthetic printed digits to mix in (0 = MNIST only)")
    ap.add_argument("--out", default=DEFAULT_WEIGHTS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    xtr, ytr, xte, yte = load_mnist()
    print(f"MNIST train {len(xtr)}, test {len(xte)}")
    if args.printed:
        t = time.time()
        xp, yp = make_printed_digits(args.printed + 2000, seed=args.seed + 1)
        xpv, ypv = xp[:2000], yp[:2000]
        xtr = np.concatenate([xtr, xp[2000:]])
        ytr = np.concatenate([ytr, yp[2000:]])
        print(f"Added {args.printed} printed digits ({time.time() - t:.0f}s)")

    model = DigitCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(xtr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(xtr))
        t, loss_sum = time.time(), 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = augment(torch.from_numpy(xtr[idx]).unsqueeze(1), rng)
            xb = (xb - MNIST_MEAN) / MNIST_STD
            loss = F.cross_entropy(model(xb), torch.from_numpy(ytr[idx]),
                                   label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
        msg = (f"epoch {ep + 1}/{args.epochs}  loss {loss_sum / len(perm):.4f}  "
               f"MNIST test acc {evaluate(model, xte, yte):.4f}")
        if args.printed:
            msg += f"  printed val acc {evaluate(model, xpv, ypv):.4f}"
        print(msg + f"  ({time.time() - t:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
