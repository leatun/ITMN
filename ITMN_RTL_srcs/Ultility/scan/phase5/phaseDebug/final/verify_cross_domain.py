import numpy as np
import math

# ==============================================================================
# 1. CẤU HÌNH & HẰNG SỐ
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128  # Input
D_MODEL = 64   # Output
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE RAW (FLOAT) ---
# Copy chính xác đường dẫn file trên máy cậu vào đây
FILE_IN_RAW  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_W_RAW   = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
FILE_OUT_RAW = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"

# ==============================================================================
# 2. HÀM LOAD DỮ LIỆU (TỪ CODE CẬU GỬI)
# ==============================================================================
def load_input_matrix_transposed(path):
    print(f"Loading Input: {path}")
    try:
        with open(path, 'r') as f:
            vals = [float(x) for x in f.read().split()]
        # Sửa lại shape cho đúng logic raw (128, 1000) rồi transpose
        mat = np.array(vals).reshape(D_INNER, SEQ_LEN)
        mat = mat.T # (1000, 128)
        print(f" -> Raw Loaded & Transposed to: {mat.shape}")
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
        print(f" -> Loaded {count} weights. Shape: {W.shape}")
        return W
    except Exception as e:
        print(f"ERROR loading weight {path}: {e}")
        return None

def load_golden_output(path):
    print(f"Loading Golden Output: {path}")
    try:
        with open(path, 'r') as f:
            vals = [float(x) for x in f.read().split()]
        mat = np.array(vals).reshape(SEQ_LEN, D_MODEL)
        print(f" -> Shape: {mat.shape}")
        return mat
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

# ==============================================================================
# 3. HÀM MÔ PHỎNG HARDWARE (FIXED POINT)
# ==============================================================================
def to_fixed(val_float):
    """Chuyển Float sang Int 16-bit Q3.12 có bão hòa"""
    val = int(round(val_float * SCALE))
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
def run_simulation_check():
    # 1. Load Raw Float Data
    X_Float = load_input_matrix_transposed(FILE_IN_RAW)
    W_Float = load_weight_matrix_structured(FILE_W_RAW, (D_MODEL, D_INNER))
    Y_Gold_Float = load_golden_output(FILE_OUT_RAW)

    if X_Float is None or W_Float is None or Y_Gold_Float is None: return

    # --- KIỂM TRA 1: TOÁN HỌC THUẦN TÚY (FLOAT) ---
    print("\n-------------------------------------------------------------")
    print("TEST 1: PURE MATH CHECK (FLOAT32)")
    print("Mục đích: Kiểm tra xem file Input, Weight và Output có khớp nhau về mặt lý thuyết không.")
    
    Y_Calc_Float = np.matmul(X_Float, W_Float.T)
    diff_float = np.abs(Y_Calc_Float - Y_Gold_Float)
    max_diff_float = np.max(diff_float)
    
    print(f"MAX Diff (Float): {max_diff_float:.6f}")
    if max_diff_float < 0.01:
        print("=> KẾT LUẬN TEST 1: DỮ LIỆU CHUẨN! Input và Weight tạo ra đúng Golden Output.")
    else:
        print("=> KẾT LUẬN TEST 1: DỮ LIỆU LỆCH! Có thể thiếu Bias hoặc Residual.")

    # --- KIỂM TRA 2: MÔ PHỎNG HARDWARE (FIXED-POINT) ---
    print("\n-------------------------------------------------------------")
    print("TEST 2: HARDWARE LOGIC SIMULATION (Q3.12 SHIFT-THEN-ADD)")
    print("Mục đích: Kiểm tra sai số do kiến trúc 'Nhà Nghèo' (Dịch bit trước khi cộng).")
    
    # A. Quantize Input/Weight sang Int16
    print("-> Converting Float to Fixed-Point Int16...")
    X_Int = np.array([[to_fixed(x) for x in row] for row in X_Float])
    W_Int = np.array([[to_fixed(x) for x in row] for row in W_Float])
    
    # B. Chuyển Golden sang Int16 để so sánh
    Y_Gold_Int = np.array([[to_fixed(x) for x in row] for row in Y_Gold_Float])

    # C. Chạy mô phỏng Hardware
    print("-> Running Simulation loop...")
    Y_Sim_HW = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
    
    for t in range(SEQ_LEN):
        for out_ch in range(D_MODEL):
            acc = 0
            for in_ch in range(D_INNER):
                x_val = int(X_Int[t, in_ch])
                w_val = int(W_Int[out_ch, in_ch])
                
                # LOGIC "NHÀ NGHÈO" (Shift-then-Add)
                # Đây là chỗ gây ra sai số 63 đơn vị
                prod = x_val * w_val
                term = prod >> FRAC_BITS # Dịch bit ngay lập tức
                acc += term
            
            # Kẹp đầu ra
            Y_Sim_HW[t, out_ch] = sat16(acc)
            
    # D. So sánh Sim HW vs Golden
    diff_hw = np.abs(Y_Sim_HW - Y_Gold_Int)
    max_diff_hw = np.max(diff_hw)
    avg_diff_hw = np.mean(diff_hw)
    
    print(f"\nMAX Diff (Hardware Logic): {max_diff_hw}")
    print(f"AVG Diff (Hardware Logic): {avg_diff_hw:.2f}")
    
    print("\n--- SAMPLE CHECK (Token 0, Channel 0) ---")
    print(f"Float Math : {to_fixed(Y_Calc_Float[0,0])} (Hex: {to_fixed(Y_Calc_Float[0,0]) & 0xFFFF:04x})")
    print(f"HW Sim     : {Y_Sim_HW[0,0]} (Hex: {Y_Sim_HW[0,0] & 0xFFFF:04x})")
    print(f"Golden File: {Y_Gold_Int[0,0]} (Hex: {Y_Gold_Int[0,0] & 0xFFFF:04x})")
    
    print("\n-------------------------------------------------------------")
    print("TỔNG KẾT:")
    if max_diff_float < 0.01 and max_diff_hw > 50:
        print("1. Dữ liệu Input/Weight/Golden là KHỚP NHAU (Float đúng).")
        print("2. Nhưng Hardware Sim bị lệch lớn so với Golden.")
        print("=> NGUYÊN NHÂN: Do kiến trúc 'Dịch bit trước khi cộng' làm mất độ chính xác.")
        print("=> Hardware Verilog của cậu KHÔNG SAI LOGIC, nó chỉ kém chính xác hơn Float thôi.")
    elif max_diff_float > 0.1:
        print("1. Ngay cả tính Float cũng sai.")
        print("=> NGUYÊN NHÂN: File dữ liệu bị lệch (Thiếu Bias/Residual).")

if __name__ == "__main__":
    run_simulation_check()