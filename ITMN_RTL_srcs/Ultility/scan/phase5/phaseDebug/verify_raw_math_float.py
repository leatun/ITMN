import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128  # Input Dimension
D_MODEL = 64   # Output Dimension

# --- ĐƯỜNG DẪN FILE RAW (FLOAT) ---
FILE_IN_RAW  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_W_RAW   = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
FILE_OUT_RAW = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"

# ==============================================================================
# HÀM LOAD DỮ LIỆU
# ==============================================================================
def load_input_matrix_transposed(path):
    print(f"Loading Input: {path}")
    try:
        with open(path, 'r') as f:
            vals = [float(x) for x in f.read().split()]
        
        # --- SỬA LOGIC TẠI ĐÂY ---
        # File Raw lưu dạng (128 Channels, 1000 Tokens)
        mat = np.array(vals).reshape(D_INNER, SEQ_LEN)
        
        # Transpose lại thành (1000 Tokens, 128 Channels) để nhân ma trận
        mat = mat.T 
        
        print(f" -> Raw Shape: ({D_INNER}, {SEQ_LEN}) -> Transposed to: {mat.shape}")
        return mat
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

def load_golden_output(path):
    print(f"Loading Golden Output: {path}")
    try:
        with open(path, 'r') as f:
            vals = [float(x) for x in f.read().split()]
        
        # Golden Output thường là (1000, 64)
        mat = np.array(vals).reshape(SEQ_LEN, D_MODEL)
        print(f" -> Shape: {mat.shape}")
        return mat
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

def load_weight_matrix_structured(path, shape):
    print(f"Loading Weight: {path}")
    W = np.zeros(shape)
    count = 0
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.replace('[', '').replace(']', '').replace(',', ' ').split()
                if len(parts) >= 3:
                    r, c = int(parts[0]), int(parts[1])
                    v = float(parts[2])
                    if r < shape[0] and c < shape[1]:
                        W[r, c] = v
                        count += 1
        print(f" -> Loaded {count} weights. Matrix Shape: {W.shape}")
        return W
    except Exception as e:
        print(f"ERROR loading weight {path}: {e}")
        return None

# ==============================================================================
# MAIN VERIFICATION
# ==============================================================================
def run_verify():
    print("=== STARTING RAW MATH VERIFICATION (FLOAT32) - CORRECTED ===\n")
    
    # 1. Load Data
    # Input X cần transpose từ (128,1000) -> (1000,128)
    X = load_input_matrix_transposed(FILE_IN_RAW) 
    W = load_weight_matrix_structured(FILE_W_RAW, (D_MODEL, D_INNER))
    Y_Gold = load_golden_output(FILE_OUT_RAW)

    if X is None or W is None or Y_Gold is None:
        return

    # 2. Tính toán Toán học (Matrix Multiplication)
    # Công thức: Y = X * W^T
    print("\n--- Calculating: Y_Calc = X @ W.T ---")
    Y_Calc = np.matmul(X, W.T)

    # 3. So sánh
    print("\n--- Comparing Y_Calc vs Y_Golden ---")
    diff = np.abs(Y_Calc - Y_Gold)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)

    print(f"MAX Diff (Float): {max_diff:.8f}")
    print(f"AVG Diff (Float): {avg_diff:.8f}")

    # 4. In mẫu kiểm tra
    print("\n--- Sample Check (Token 0, Channel 0) ---")
    print(f"Calc: {Y_Calc[0,0]:.6f}")
    print(f"Gold: {Y_Gold[0,0]:.6f}")
    print(f"Diff: {diff[0,0]:.6f}")

    # 5. Check Bias (Nếu Diff lớn)
    # Bias = Gold - Calc (Trung bình)
    avg_bias = np.mean(Y_Gold - Y_Calc, axis=0)
    print("\n--- Estimated Bias (First 5 channels) ---")
    print(avg_bias[:5])

if __name__ == "__main__":
    run_verify()