import numpy as np
import math

# --- CẤU HÌNH ---
IN_DIM = 128
OUT_DIM = 36 # 4 dt + 16 B + 16 C
FRAC_BITS = 12
SCALE = 1 << 12
MAX_INT = 32767
MIN_INT = -32768

# File gốc (File chứa [row, col, val])
INPUT_FILE = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/x_proj_weight.txt"
# File Output (File Hex nạp cho TB)
OUTPUT_FILE = "w_xproj_reordered.txt"

def float_to_hex(val):
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def run():
    print("--- REORDERING X_PROJ WEIGHTS ---")
    
    # 1. Load Matrix Gốc (36, 128)
    w_orig = np.zeros((OUT_DIM, IN_DIM))
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                parts = line.strip().replace('[','').replace(']','').replace(',',' ').split()
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    if r < OUT_DIM and c < IN_DIM:
                        w_orig[r, c] = v
    except Exception as e:
        print(f"Error reading: {e}"); return

    # 2. Tạo Matrix Mới (Theo thứ tự Hardware muốn)
    # Hardware muốn: Chunk 0 là B, Chunk 1 là C, Chunk 2 là dt
    w_new = np.zeros((48, 128)) # 48 dòng (3 chunks * 16)
    
    # Map B (Gốc 4-19) -> Mới 0-15
    print("Mapping B (Orig 4-19) -> New Chunk 0 (0-15)")
    w_new[0:16, :] = w_orig[4:20, :]
    
    # Map C (Gốc 20-35) -> Mới 16-31
    print("Mapping C (Orig 20-35) -> New Chunk 1 (16-31)")
    w_new[16:32, :] = w_orig[20:36, :]
    
    # Map dt (Gốc 0-3) -> Mới 32-35
    print("Mapping dt (Orig 0-3) -> New Chunk 2 (32-35)")
    w_new[32:36, :] = w_orig[0:4, :]
    
    # Các dòng 36-47 để là 0.0 (Padding)
    
    # 3. Xuất file Hex theo cấu trúc Hardware đọc
    # Loop Chunk -> Loop Col -> Loop Row16
    hex_lines = []
    num_chunks = 3
    
    for chunk in range(num_chunks):
        start_row = chunk * 16
        for col in range(IN_DIM):
            for i in range(16):
                r = start_row + i
                val = w_new[r, col]
                hex_lines.append(float_to_hex(val))
                
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(hex_lines))
        
    print(f"DONE! Saved to {OUTPUT_FILE}")
    print(f"Total lines: {len(hex_lines)}")

if __name__ == "__main__":
    run()