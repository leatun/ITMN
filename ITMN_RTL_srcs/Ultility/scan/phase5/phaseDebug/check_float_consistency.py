import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128 # Input
D_MODEL = 64  # Output

# --- FILE FLOAT GỐC (Sửa đường dẫn của cậu vào đây) ---
FILE_IN  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_W   = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
FILE_OUT = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"

def run_check():
    print("--- 1. Loading FLOAT Files ---")
    
    # 1. Load Input
    try:
        with open(FILE_IN, 'r') as f:
            vals = [float(x) for x in f.read().split()]
        X = np.array(vals).reshape(SEQ_LEN, D_INNER)
        print(f"Input Loaded: {X.shape}")
        print(f"Sample X[0,0]: {X[0,0]}")
    except Exception as e: print(f"Err X: {e}"); return

    # 2. Load Weight (Parse kỹ index [r,c])
    W = np.zeros((D_MODEL, D_INNER))
    try:
        with open(FILE_W, 'r') as f:
            count = 0
            for line in f:
                # Format: "[row,col] val"
                parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    if r < D_MODEL and c < D_INNER:
                        W[r, c] = v
                        count += 1
        print(f"Weight Loaded: {W.shape}, Non-zero/entries read: {count}")
        print(f"Sample W[0,0]: {W[0,0]}")
    except Exception as e: print(f"Err W: {e}"); return

    # 3. Load Golden Output
    try:
        with open(FILE_OUT, 'r') as f:
            vals = [float(x) for x in f.read().split()]
        Y_Gold = np.array(vals).reshape(SEQ_LEN, D_MODEL)
        print(f"Golden Loaded: {Y_Gold.shape}")
        print(f"Sample Gold[0,0]: {Y_Gold[0,0]}")
    except Exception as e: print(f"Err Gold: {e}"); return

    # --- 2. CALCULATE IN FLOAT (Standard Matrix Mul) ---
    print("\n--- 2. Calculating X * W^T (Float32) ---")
    # Y = X . W.T
    Y_Calc = np.matmul(X, W.T)
    
    print(f"Sample Calc[0,0]: {Y_Calc[0,0]}")

    # --- 3. COMPARE ---
    print("\n--- 3. Verdict ---")
    diff = np.abs(Y_Calc - Y_Gold)
    max_diff = np.max(diff)
    
    print(f"MAX Diff (Float): {max_diff:.6f}")
    
    if max_diff < 1e-4:
        print(">>> FILES ARE CONSISTENT (Khớp nhau).")
        print("Vấn đề nằm ở khâu Convert sang Fixed-Point (nhân với 4096 bị sai/tràn).")
    else:
        print(">>> FILES ARE INCONSISTENT (Không khớp).")
        print("Dữ liệu Input, Weight và Output KHÔNG thuộc về cùng một phép tính Linear.")
        print("Khả năng cao:")
        print("1. File Weight này không phải của layer này.")
        print("2. File Output chứa cả Residual (Input + Linear).")
        
        # Test giả thuyết Residual (Cộng Input Phase 1 nếu có)
        # (Ở đây ta không có file Input Phase 1 float, nhưng nếu diff lớn thì chắc chắn là lệch file)

if __name__ == "__main__":
    run_check()