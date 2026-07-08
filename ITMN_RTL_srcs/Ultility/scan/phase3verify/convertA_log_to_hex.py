import numpy as np
import re

# --- CẤU HÌNH ---
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

INPUT_FILE = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/A_log.txt"
OUTPUT_FILE = "A_log_input.txt" # Nạp cái này vào TB

def float_to_hex(val):
    v = int(round(val * SCALE))
    if v > MAX_INT: v = MAX_INT
    if v < MIN_INT: v = MIN_INT
    if v < 0: v = (1 << 16) + v
    return f"{v & 0xFFFF:04x}"

def run():
    print("Converting A_log to Hex (Q3.12)...")
    # A_log shape (128, 16)
    # Hardware lưu: 128 dòng, mỗi dòng 16 phần tử (256-bit)
    matrix = np.zeros((128, 16))
    
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    matrix[r, c] = v
    except Exception as e:
        print(f"Error: {e}")
        return

    hex_lines = []
    for r in range(128):
        # Gom 16 phần tử của hàng r thành 1 dòng hex (cho dễ nạp DMA)
        # Hoặc viết từng dòng 16-bit nếu dùng $readmemh đơn giản
        # Ở đây tôi viết từng dòng 16-bit để khớp với logic các file trước
        for c in range(16):
            hex_lines.append(float_to_hex(matrix[r, c]))
            
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(hex_lines))
    print(f"DONE! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()