#!/usr/bin/env python3
import math
from pathlib import Path

# ====== HARD-CODE PATH HERE ======
#FILE_PATH = Path(r"cpp_golden_files/06_06_MambaBlock_input.txt")
#FILE_PATH = Path(r"cpp_golden_files/07_07_MambaBlock_after_norm.txt")
#FILE_PATH = Path(r"../rmsnor/rms_real_input.txt")
# =================================

def find_min_max(path: Path):
    min_val = float("inf")
    max_val = float("-inf")
    min_pos = None  # (line, col, idx)
    max_pos = None

    count = 0
    skipped = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            for col_no, tok in enumerate(parts, start=1):
                tok = tok.strip().strip(",")

                try:
                    x = float(tok)
                except ValueError:
                    skipped += 1
                    continue

                if math.isnan(x):
                    skipped += 1
                    continue

                count += 1
                if x < min_val:
                    min_val = x
                    min_pos = (line_no, col_no, count)
                if x > max_val:
                    max_val = x
                    max_pos = (line_no, col_no, count)

    if count == 0:
        raise RuntimeError("Không đọc được số nào hết (file trống hoặc format lạ).")

    return count, skipped, min_val, min_pos, max_val, max_pos


def main():
    # Đảm bảo path tính theo vị trí file script (đỡ phụ thuộc cwd)
    script_dir = Path(__file__).resolve().parent
    path = (script_dir / FILE_PATH).resolve() if not FILE_PATH.is_absolute() else FILE_PATH

    if not path.exists():
        raise FileNotFoundError(f"Không thấy file: {path}")

    count, skipped, mn, mn_pos, mx, mx_pos = find_min_max(path)
    mn_line, mn_col, mn_idx = mn_pos
    mx_line, mx_col, mx_idx = mx_pos

    print("==== RESULT ====")
    print(f"File         : {path}")
    print(f"Total numbers: {count}")
    print(f"Skipped token: {skipped}")
    print(f"MIN = {mn:.17g}  at line {mn_line}, col {mn_col}, index {mn_idx}")
    print(f"MAX = {mx:.17g}  at line {mx_line}, col {mx_col}, index {mx_idx}")


if __name__ == "__main__":
    main()
