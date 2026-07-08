import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
FRAC_BITS = 12
MAX_INT = 32767
MIN_INT = -32768

# --- INPUT FILES (Phải là file cậu đang nạp cho TB) ---
F_INPUT_CHUNKED = "x_prime_chunked_verified.txt" 
F_WEIGHT_REORDERED = "w_xproj_reordered.txt"

# --- OUTPUT FILES (Dùng cái này để Verify trong TB) ---
F_OUT_B_HW = "gold_B_hw.txt"
F_OUT_C_HW = "gold_C_hw.txt"

# --- REFERENCE FILES (Của PTB để so sánh sai số) ---
F_REF_B = "scan_real_B_shared.txt"
F_REF_C = "scan_real_C_shared.txt"

# ==============================================================================
# HÀM XỬ LÝ BIT-EXACT
# ==============================================================================
def hex_to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def to_hex(val):
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def hard_mac_calculation(input_vec, weight_matrix):
    """
    Mô phỏng chính xác Unified_PE.v:
    1. Nhân (32-bit)
    2. Dịch bit NGAY LẬP TỨC (Quantization)
    3. Cộng vào Accumulator (16-bit)
    4. Saturation (Kẹp) NGAY LẬP TỨC sau mỗi phép cộng
    """
    out_dim = len(weight_matrix)
    output = []
    
    # Giới hạn của thanh ghi Accumulator 16-bit
    ACC_MAX = 32767
    ACC_MIN = -32768
    
    for r in range(out_dim):
        acc = 0 # acc_reg khởi tạo = 0
        w_row = weight_matrix[r]
        
        for i in range(128):
            # 1. Multiply (Logic: mult_raw = in_A * in_B)
            # Python tự động xử lý số lớn, nhưng ta mô phỏng logic signed
            prod = int(input_vec[i]) * int(w_row[i])
            
            # 2. Immediate Shift (Logic: mult_shifted = mult_raw >>> 12)
            # Trong Verilog số âm >>> 12 vẫn giữ dấu, Python >> 12 cũng vậy.
            term = prod >> FRAC_BITS
            
            # 3. Accumulate (Logic: temp_result = acc_reg + mult_shifted)
            # Lưu ý: temp_result trong Verilog là 32-bit nên chưa tràn ngay
            temp_acc = acc + term
            
            # 4. Saturation Immediate (Logic: Update acc_reg)
            # Vì acc_reg chỉ có 16-bit, ta phải kẹp ngay sau mỗi chu kỳ
            if temp_acc > ACC_MAX:
                acc = ACC_MAX
            elif temp_acc < ACC_MIN:
                acc = ACC_MIN
            else:
                acc = temp_acc
        
        # Kết thúc loop 128, acc chính là giá trị cuối cùng trong PE
        output.append(acc)
        
    return output

# ==============================================================================
# LOAD DỮ LIỆU (Tái tạo lại Ma trận từ file nạp TB)
# ==============================================================================
def load_data():
    print("Loading Inputs & Weights...")
    
    # 1. Load Input (Chunked -> Token-First Matrix)
    # File: Loop Chunk(8) -> Loop Token(1000) -> Loop Ch16
    with open(F_INPUT_CHUNKED) as f: 
        lines_x = [l.strip() for l in f if l.strip()]
        
    X_matrix = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    idx = 0
    for chunk in range(8):
        base_ch = chunk * 16
        for t in range(SEQ_LEN):
            for i in range(16):
                val = hex_to_signed(lines_x[idx])
                X_matrix[t][base_ch + i] = val
                idx += 1
                
    # 2. Load Weight (Reordered -> Rows Matrix)
    # File: Loop Chunk(3) -> Loop Col(128) -> Loop Row16
    with open(F_WEIGHT_REORDERED) as f:
        lines_w = [l.strip() for l in f if l.strip()]
        
    W_matrix = np.zeros((48, 128), dtype=int) # 48 rows (do padding)
    idx = 0
    for chunk in range(3):
        start_row = chunk * 16
        for col in range(128):
            for i in range(16):
                val = hex_to_signed(lines_w[idx])
                W_matrix[start_row + i][col] = val
                idx += 1
                
    return X_matrix, W_matrix

# ==============================================================================
# MAIN RUN
# ==============================================================================
def run():
    X, W = load_data()
    print("Data Loaded. Starting Simulation...")
    
    B_hw = []
    C_hw = []
    
    # --- STATISTICS ---
    max_diff_B = 0
    max_diff_C = 0
    
    # Load Ref để so sánh luôn
    with open(F_REF_B) as f: ref_B = [hex_to_signed(l) for l in f if l.strip()]
    with open(F_REF_C) as f: ref_C = [hex_to_signed(l) for l in f if l.strip()]
    
    # --- LOOP SIMULATION ---
    for t in range(SEQ_LEN):
        x_vec = X[t]
        
        # Tính toán (Full 48 rows, nhưng ta chỉ quan tâm B và C)
        # Chunk 0 (Rows 0-15): B
        # Chunk 1 (Rows 16-31): C
        
        # Calc B
        row_B = hard_mac_calculation(x_vec, W[0:16])
        # Calc C
        row_C = hard_mac_calculation(x_vec, W[16:32])
        
        # Lưu vào list để xuất file
        B_hw.extend(row_B)
        C_hw.extend(row_C)
        
        # Compare with Reference (Just for Info)
        base_idx = t * 16
        for i in range(16):
            diff_b = abs(row_B[i] - ref_B[base_idx + i])
            diff_c = abs(row_C[i] - ref_C[base_idx + i])
            max_diff_B = max(max_diff_B, diff_b)
            max_diff_C = max(max_diff_C, diff_c)

    # --- EXPORT FILES ---
    print(f"Writing {F_OUT_B_HW}...")
    with open(F_OUT_B_HW, 'w') as f:
        for val in B_hw: f.write(to_hex(val) + "\n")
        
    print(f"Writing {F_OUT_C_HW}...")
    with open(F_OUT_C_HW, 'w') as f:
        for val in C_hw: f.write(to_hex(val) + "\n")
        
    print("\n========================================")
    print(f"HARDWARE GOLDEN GENERATED!")
    print(f"Max Diff vs PyTorch (B): {max_diff_B}")
    print(f"Max Diff vs PyTorch (C): {max_diff_C}")
    print("========================================")
    print("HÃY DÙNG FILE 'gold_B_hw.txt' VÀ 'gold_C_hw.txt' TRONG TESTBENCH!")
    print("Nếu Verilog khớp với file này -> Hardware ĐÚNG.")
    print("Sai số trên là do giới hạn của Fixed-Point.")

if __name__ == "__main__":
    run()