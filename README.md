# AR-Sudoku-Solver

Implementation of an AR-powered Sudoku Solver. Point a webcam at a printed
sudoku and the missing digits are drawn back onto the live video, locked to
the paper as it moves.

Runs on Windows, macOS and Linux, on CPU (no GPU needed), with Python 3.9 or newer.

## Quick start

**macOS / Linux**
```bash
cd AR-Sudoku-Solver
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
python ar_sudoku.py
```

**Windows (PowerShell or Command Prompt)**
```bat
cd AR-Sudoku-Solver
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py
python ar_sudoku.py
```

If PowerShell blocks `activate`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.

Pretrained weights are included at `weights/mnist_cnn.pt`, so training is
optional. `check_setup.py` checks the libraries, the model, the solver, the
full pipeline on synthetic photos and whether a window can open, and
`python check_setup.py --camera` also tests the webcam.

**Linux without a GPU:** `pip install torch` downloads the large CUDA build.
For a much smaller CPU build, install torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`, then
`pip install -r requirements.txt`.

## Usage

```bash
python ar_sudoku.py                                    # webcam 0
python ar_sudoku.py --source 1                         # another camera
python ar_sudoku.py --save                             # record the webcam session
python ar_sudoku.py --source puzzle.jpg --save         # output/solved_<date>_<time>.jpg
python ar_sudoku.py --source clip.mp4 --save my_run    # output/my_run.mp4
python ar_sudoku.py --no-show --save                   # no window (SSH, server, Docker)
python ar_sudoku.py --device cuda                      # auto, cpu, cuda or mps
```

Keys in the window: `q` or `Esc` quit, `r` reset the reading (after switching
puzzles), `s` save a snapshot. Closing the window also quits.

### Where files are saved

Everything the project saves goes into the `output/` folder, which is
created automatically and listed in `.gitignore`, so results never end up in
git:

```
output/
├── solved_20261016_153226.jpg   # --save with no name: timestamped, never overwritten
├── my_run.mp4                   # --save my_run
├── snapshots/                   # s key during a live session
├── demo.mp4, demo_frame.jpg     # tests/make_demo_video.py
└── examples/                    # tests/test_pipeline.py --save-examples
```

A name passed to `--save` is placed inside `output/` (subfolders such as
`--save runs/a.mp4` work too). An absolute path is used exactly as given.

Hold the puzzle so it fills about half the frame. A blue outline shows when
the grid is found, and the solution appears in green after about 6 frames.

All scripts work from any folder, since paths resolve relative to the project.

## Pipeline

1. **Grid detection** (`grid_detector.find_grid`): adaptive threshold, nested
   contours, convex hull, 4 point polygon. Among large square-ish quads the
   smallest one with dark box lines at 1/3 and 2/3 is kept, so the paper edge
   is never mistaken for the grid.
2. **Rectification**: a homography maps the grid to a 504 x 504 board
   (56 px cells).
3. **Digit extraction** (`digit_model.extract_digit`): each cell is
   thresholded, border line remnants are cleared, the central connected
   component is kept (or the cell is marked empty), then it is converted to
   MNIST format: 20 x 20 box inside 28 x 28, centred by centre of mass.
4. **Recognition**: a CNN trained on MNIST classifies all non empty cells in
   one batch. Class 0 is masked out since sudoku uses 1 to 9.
5. **Temporal voting**: readings from the last 6 frames are combined per cell
   by majority vote, which removes single frame misreads.
6. **Solving** (`solver.py`): bitmask backtracking with the minimum remaining
   values heuristic. If the read board is invalid or unsolvable, the least
   confident givens are swapped for their second best class until it solves.
   Each distinct board is solved once and cached.
7. **AR overlay**: missing digits are drawn in rectified board space and
   warped back with the inverse homography, so they follow perspective.

Backtracking was chosen over RL: sudoku is a pure constraint satisfaction
problem where exact search is guaranteed correct and takes milliseconds,
whereas an RL policy would need a huge state space, sparse rewards and still
offer no correctness guarantee.

## Training (optional)

```bash
python download_mnist.py               # ~55 MB into data/MNIST/raw
python train_mnist.py --printed 20000  # MNIST + synthetic printed digits, ~15 min on CPU
python train_mnist.py                  # MNIST only
```

Printed digits are rendered from the fonts installed on your machine (Windows,
macOS and Linux font folders are all searched) and passed through the same
cell preprocessing as live frames. Training overwrites `weights/mnist_cnn.pt`
unless `--out` is given.

## Results

Measured on CPU, 5 epochs each:

| Model | MNIST test acc | Held out printed digits |
|---|---|---|
| MNIST only | 99.65% | not measured |
| MNIST + 20,000 printed (`weights/mnist_cnn.pt`) | 99.64% | 99.90% |

End to end from a **single frame**, on synthetic camera photos (perspective,
lighting gradient, blur, noise, JPEG compression, clutter):

| Metric (200 photos) | MNIST only | MNIST + printed |
|---|---|---|
| Grid found | 200/200 | 200/200 |
| Cell accuracy | 98.83% | 99.67% |
| Digit cells correct | 97.43% | 99.70% |
| Boards read perfectly | 117/200 | 187/200 |
| Correctly solved (with repair) | 148/200 | 193/200 |

On 100 photos printed in OpenCV Hershey fonts, never seen in training, the
included model solved 99/100. Grid corners are located to within 0.11 of a
cell. Live video adds 6 frame voting on top of these single frame numbers.
Processing runs at about 18 fps on a 2 core cloud CPU.

Solver speed (`python tests/bench_solver.py`, pure Python, median of 20 runs,
measured on a laptop CPU):

| Puzzle | Clues | Solve | Solve + uniqueness check |
|---|---|---|---|
| Easy | 30 | 0.08 ms | 0.08 ms |
| Medium | 36 | 0.04 ms | 0.05 ms |
| Hard (17 clues) | 17 | 1.47 ms | 7.90 ms |
| Inkala "world's hardest" | 21 | 20.6 ms | 36.2 ms |

The solver is never the bottleneck: each distinct board is solved once and
the solution is cached, so per frame time goes to grid detection and digit
reading.

The MNIST only model mostly misreads printed 1 as 2 and 6 as 8, which is why
the printed digit mix in is used for the included weights.

## Tests

```bash
python tests/test_pipeline.py --n 200                    # accuracy on synthetic camera photos
python tests/test_pipeline.py --n 50 --save-examples     # also saves 6 images to output/examples
python tests/make_demo_video.py                          # simulated handheld video, output/demo.mp4
```

## Troubleshooting

| Problem | Fix |
|---|---|
| macOS: "Cannot open camera" | System Settings > Privacy & Security > Camera, allow your terminal or IDE, then restart it |
| Windows: camera fails or opens slowly | Close other apps using it (Teams, Zoom); check Settings > Privacy & security > Camera |
| Linux: "Cannot open camera" | `sudo usermod -aG video $USER`, then log out and in |
| Wrong camera opens (e.g. iPhone Continuity Camera) | `--source 1` |
| "No display found" | Expected over SSH or on servers; use `--no-show --save` and open the video in `output/` |
| Linux: `libGL.so.1` missing | `sudo apt install libgl1`, or use `opencv-python-headless` with `--no-show` |
| MNIST download SSL error on macOS | `pip install certifi` (in requirements), or run "Install Certificates.command" in your Python folder |
| Saved as `.avi` instead of `.mp4` | Your OpenCV build lacks the mp4 encoder; the `.avi` plays everywhere |
| Stuck at "no valid solution yet" | One digit is misread; even the lighting, avoid glare, hold steadier and press `r` |
