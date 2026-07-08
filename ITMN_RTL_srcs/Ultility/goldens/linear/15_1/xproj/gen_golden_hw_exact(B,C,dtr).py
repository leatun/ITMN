import numpy as np

# ==============================================================================
# CẤU HÌNH HỆ THỐNG
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
FRAC_BITS = 12
MAX_INT = 32767
MIN_INT = -32768

# --- INPUT FILES ---
# 1. File Input của Linear (Output của Conv mà cậu đã verify)
F_INPUT_CHUNKED = "x_prime_chunked_verified.txt" 
# 2. File Weight đã Reorder (Chunk 0=B, Chunk 1=C, Chunk 2=dt)
F_WEIGHT_REORDERED = "w_xproj_reordered.txt"

# --- OUTPUT FILES (Golden Hardware) ---
F_OUT_B_HW = "gold_B_hw.txt"
F_OUT_C_HW = "gold_C_hw.txt"
F_OUT_DT_RAW_HW = "gold_dt_raw_hw.txt" # <--- Cái cậu đang cần

# ==============================================================================
# 1. HARDWARE-ACCURATE MAC UNIT
# ==============================================================================
def hex_to_signed(hex_str):
    """Chuyển Hex 16-bit sang Signed Int"""
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def to_hex(val):
    """Chuyển Signed Int sang Hex 16-bit"""
    if val > MAX_INT: val = MAX_INT     # Safety clamp output
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def hard_mac_calculation(input_vec, weight_matrix):
    """
    Mô phỏng chính xác logic Unified_PE.v:
    Acc = Saturation( Acc + ( (Input * Weight) >>> 12 ) )
    """
    out_dim = len(weight_matrix)
    output = []
    
    # Range Accumulator 16-bit
    ACC_MAX = 32767
    ACC_MIN = -32768
    
    for r in range(out_dim):
        acc = 0 # Reset Accumulator mỗi hàng
        w_row = weight_matrix[r]
        
        for i in range(128): # D_INNER Loop
            # 1. Nhân (Signed)
            prod = int(input_vec[i]) * int(w_row[i])
            
            # 2. Dịch bit ngay lập tức (Immediate Quantization)
            # Python '>>' với số âm hoạt động chuẩn như Verilog '>>>'
            term = prod >> FRAC_BITS
            
            # 3. Cộng dồn
            temp_acc = acc + term
            
            # 4. Kẹp (Saturation) ngay lập tức sau mỗi phép cộng
            if temp_acc > ACC_MAX:
                acc = ACC_MAX
            elif temp_acc < ACC_MIN:
                acc = ACC_MIN
            else:
                acc = temp_acc
        
        output.append(acc)
    return output

# ==============================================================================
# 2. DATA LOADER
# ==============================================================================
def load_data():
    print("Loading Inputs & Weights...")
    
    # Load Input X (Format: 8 Chunks * 1000 Tokens * 16 Channels)
    with open(F_INPUT_CHUNKED) as f: 
        lines_x = [l.strip() for l in f if l.strip()]
        
    X_matrix = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    idx = 0
    # Reconstruct Token-First Matrix from Chunked File
    for chunk in range(8):
        base_ch = chunk * 16
        for t in range(SEQ_LEN):
            for i in range(16):
                X_matrix[t][base_ch + i] = hex_to_signed(lines_x[idx])
                idx += 1
                
    # Load Weight (Format: 3 Chunks * 128 Cols * 16 Rows)
    with open(F_WEIGHT_REORDERED) as f:
        lines_w = [l.strip() for l in f if l.strip()]
        
    W_matrix = np.zeros((48, 128), dtype=int) # 48 rows (3 chunks)
    idx = 0
    for chunk in range(3):
        start_row = chunk * 16
        for col in range(128):
            for i in range(16):
                W_matrix[start_row + i][col] = hex_to_signed(lines_w[idx])
                idx += 1
                
    return X_matrix, W_matrix

# ==============================================================================
# 3. MAIN GENERATOR
# ==============================================================================
def run():
    X, W = load_data()
    print("Generating Hardware Golden Files...")
    
    # Weight Map (Dựa trên reorder script):
    # Chunk 0 (Rows 0-15)  -> B
    # Chunk 1 (Rows 16-31) -> C
    # Chunk 2 (Rows 32-47) -> dt_raw (Chỉ 4 row đầu valid)
    
    W_B = W[0:16]
    W_C = W[16:32]
    W_DT = W[32:48]
    
    list_B = []
    list_C = []
    list_DT = []
    
    for t in range(SEQ_LEN):
        x_vec = X[t]
        
        # --- CALC B ---
        res_b = hard_mac_calculation(x_vec, W_B)
        list_B.extend(res_b)
        
        # --- CALC C ---
        res_c = hard_mac_calculation(x_vec, W_C)
        list_C.extend(res_c)
        
        # --- CALC DT_RAW ---
        # Tính toán cả 16 hàng (để mô phỏng PE chạy hết)
        res_dt_full = hard_mac_calculation(x_vec, W_DT)
        
        # Apply Zero-Padding Logic (Như Verilog Controller)
        # Giữ 4 số đầu, 12 số sau ép về 0
        res_dt_clean = res_dt_full[0:4] + [0]*12
        
        list_DT.extend(res_dt_clean)

    # --- WRITE FILES ---
    def write_file(fname, data_list):
        print(f"Writing {fname}...")
        with open(fname, 'w') as f:
            for val in data_list:
                f.write(to_hex(val) + "\n")

    write_file(F_OUT_B_HW, list_B)
    write_file(F_OUT_C_HW, list_C)
    write_file(F_OUT_DT_RAW_HW, list_DT)
    
    print("\nDONE! Files generated:")
    print(f"1. {F_OUT_B_HW} (Valid B)")
    print(f"2. {F_OUT_C_HW} (Valid C)")
    print(f"3. {F_OUT_DT_RAW_HW} (Valid dt[0:4] + Zeros)")
    print("\nTip: Load file này vào Testbench và so sánh với RAM tại ADDR_DT_RAW_BASE.")

if __name__ == "__main__":
    run()