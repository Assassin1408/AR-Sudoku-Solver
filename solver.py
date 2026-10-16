"""Backtracking sudoku solver with bitmask constraints and the minimum
remaining values (MRV) heuristic. Typical newspaper puzzles solve in a few
milliseconds; Arto Inkala's "world's hardest" puzzle takes ~100 ms in pure
Python, including the uniqueness check."""
import numpy as np

ALL = 0x3FE  # bits 1..9


def _box(r, c):
    return (r // 3) * 3 + c // 3


def is_valid_givens(board):
    """True if no row, column or box repeats a digit."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for r in range(9):
        for c in range(9):
            d = int(board[r][c])
            if d:
                bit = 1 << d
                b = _box(r, c)
                if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                    return False
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
    return True


def solve(board, count_limit=1):
    """board: 9x9 ints, 0 = empty. Returns (solution as np.ndarray or None,
    number of solutions found up to count_limit). Use count_limit=2 to check
    uniqueness."""
    g = [[int(v) for v in row] for row in board]
    if not is_valid_givens(g):
        return None, 0
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empties = []
    for r in range(9):
        for c in range(9):
            d = g[r][c]
            if d:
                bit = 1 << d
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box(r, c)] |= bit
            else:
                empties.append((r, c, _box(r, c)))

    found = []

    def bt():
        if not empties:
            found.append([row[:] for row in g])
            return len(found) >= count_limit
        # MRV: pick the empty cell with the fewest candidates
        best_i, best_cand, best_n = -1, 0, 10
        for i, (r, c, b) in enumerate(empties):
            cand = ALL & ~(rows[r] | cols[c] | boxes[b])
            n = bin(cand).count("1")
            if n < best_n:
                best_i, best_cand, best_n = i, cand, n
                if n <= 1:
                    break
        if best_n == 0:
            return False
        r, c, b = empties.pop(best_i)
        cand = best_cand
        while cand:
            bit = cand & -cand
            cand ^= bit
            g[r][c] = bit.bit_length() - 1
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if bt():
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        g[r][c] = 0
        empties.insert(best_i, (r, c, b))
        return False

    bt()
    return (np.array(found[0]) if found else None), len(found)


if __name__ == "__main__":
    import time
    p = ("8........" "..36....." ".7..9.2.." ".5...7..." "....457.."
         "...1...3." "..1....68" "..85...1." ".9....4..")  # Arto Inkala "hardest"
    b = [[int(ch) if ch != "." else 0 for ch in p[i * 9:i * 9 + 9]] for i in range(9)]
    t = time.perf_counter()
    sol, n = solve(b, count_limit=2)
    print(sol, "\nsolutions:", n, f"time {1000 * (time.perf_counter() - t):.1f} ms")
