import numpy as np

# --- CẤU HÌNH ---
D_INNER = 128
SCALE = 1 << 12
INPUT_FILE = "x_prime_chunked_for_tb.txt" # File input cậu đã verify OK
WEIGHT_FILE = "x_proj_weight_ptb.txt"     # File weight hiện tại
GOLD_B = "scan_real_B_shared.txt"
GOLD_C = "scan_real_C_shared.txt"
GOLD_DT = "scan_real_delta.txt"           # Dùng delta để check dt (dù delta đã qua softplus)

def hex_to_int(h):
    v = int(h, 16)
    return v - 65536 if v & 0x8000 else v

def run_debug():
    print("--- DEBUGGING X_PROJ ROW ORDER ---")
    
    # 1. Load Input Token 0 (128 values)
    # File chunked: Dòng 0..15 là 16 kênh đầu của Token 0.
    # Dòng 16..31 là 16 kênh đầu của Token 1 -> KHÔNG PHẢI.
    # File chunked: Chunk 0 (1000 dòng) -> Dòng 0 là T0.
    # Chunk 1 (1000 dòng) -> Dòng 1000 là T0.
    
    with open(INPUT_FILE) as f: lines = [l.strip() for l in f if l.strip()]
    
    x_vec = []
    for chunk in range(8):
        # Lấy dòng đầu tiên của mỗi Chunk (ứng với Token 0)
        line_idx = chunk * 1000 
        val_hex = lines[line_idx] 
        # File chunked lưu 1 dòng hex (16bit) hay pack? 
        # Script tạo file chunked lưu từng dòng hex.
        # Nhưng đợi đã! Script tạo file chunked loop: Chunk -> Token -> 16 Channel
        # Vậy 16 dòng đầu tiên là 16 channel của Token 0 (Chunk 0).
        
        # Logic load lại cho đúng script input:
        start_idx = chunk * (1000 * 16) # Mỗi chunk có 1000 token * 16 dòng
        # 16 dòng đầu của chunk là Token 0
        for i in range(16):
            x_vec.append(hex_to_int(lines[start_idx + i]))
            
    x_vec = np.array(x_vec) # Shape (128,)
    print(f"Loaded Input Token 0: {x_vec[:5]} ...")

    # 2. Load Weight (48 rows do padding, 128 cols)
    # File weight cậu tạo từ script padding: Chunk -> Col -> Row
    with open(WEIGHT_FILE) as f: w_lines = [l.strip() for l in f if l.strip()]
    
    W_matrix = np.zeros((48, 128), dtype=int)
    idx = 0
    for chunk in range(3):
        start_row = chunk * 16
        for col in range(128):
            for r in range(16):
                if idx < len(w_lines):
                    W_matrix[start_row + r, col] = hex_to_int(w_lines[idx])
                    idx += 1
                    
    # 3. Tính toán 36 Output đầu tiên (Full precision -> Shift)
    calc_out = []
    for r in range(36):
        acc = np.sum(x_vec * W_matrix[r])
        res = int(acc) >> 12
        calc_out.append(res)
        
    print("\n--- CALCULATED OUTPUT (Rows 0-35) ---")
    print(f"Rows 00-03: {calc_out[0:4]}")
    print(f"Rows 04-19: {calc_out[4:20]}")
    print(f"Rows 20-35: {calc_out[20:36]}")

    # 4. Load Golden (Token 0)
    with open(GOLD_B) as f: b_gold = [hex_to_int(f.readline().strip()) for _ in range(16)]
    with open(GOLD_C) as f: c_gold = [hex_to_int(f.readline().strip()) for _ in range(16)]
    
    print("\n--- GOLDEN EXPECTATION ---")
    print(f"Golden B (First 16): {b_gold[:5]} ...")
    print(f"Golden C (First 16): {c_gold[:5]} ...")
    
    # 5. SO SÁNH
    print("\n--- MATCHING ANALYSIS ---")
    
    # Check if Rows 0-3 match B?
    print(f"Check: Calc Row 0 vs Gold B[0]: {calc_out[0]} vs {b_gold[0]}")
    
    # Check if Rows 4-19 match B?
    print(f"Check: Calc Row 4 vs Gold B[0]: {calc_out[4]} vs {b_gold[0]}")
    
    # Check if Rows 0-15 match B?
    print(f"Check: Calc Row 0 vs Gold B[0]: {calc_out[0]} vs {b_gold[0]}")

if __name__ == "__main__":
    run_debug()