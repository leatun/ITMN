import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_MODEL = 64
FRAC_BITS = 11
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- FILES HEX (Đã tạo từ trước) ---
FILE_HEX_IN     = "debug_input_correct.txt"
FILE_HEX_W      = "debug_weight_correct.txt"
FILE_HEX_GOLD   = "debug_golden_correct.txt"

# --- FILES RAW FLOAT (Gốc) ---
FILE_RAW_IN     = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_RAW_W      = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
FILE_RAW_OUT    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"

# Ngưỡng kích hoạt soi lỗi
TRIGGER_DIFF = 800

# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str.strip(), 16)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def to_fixed(val_float):
    return int(round(val_float * SCALE))

# ==============================================================================
# LOAD DATA (HEX & RAW)
# ==============================================================================
def load_all_data():
    print("--- Loading HEX Data ---")
    with open(FILE_HEX_IN) as f: 
        X_Hex = np.array([to_signed(l) for l in f]).reshape(SEQ_LEN, D_INNER)
    with open(FILE_HEX_GOLD) as f:
        Y_Gold_Hex = np.array([to_signed(l) for l in f]).reshape(SEQ_LEN, D_MODEL)
    
    # Reconstruct W Hex
    with open(FILE_HEX_W) as f: w_flat = [to_signed(l) for l in f]
    W_Hex = np.zeros((D_MODEL, D_INNER), dtype=int)
    idx = 0
    for chunk in range(4):
        for col in range(128):
            for r in range(16):
                if idx < len(w_flat): W_Hex[chunk*16+r, col] = w_flat[idx]; idx+=1

    print("--- Loading RAW FLOAT Data ---")
    # Input Raw (Transpose)
    with open(FILE_RAW_IN) as f: vals = [float(x) for x in f.read().split()]
    X_Float = np.array(vals).reshape(D_INNER, SEQ_LEN).T
    
    # Weight Raw
    W_Float = np.zeros((D_MODEL, D_INNER))
    with open(FILE_RAW_W) as f:
        for line in f:
            parts = line.replace('[','').replace(']','').replace(',',' ').split()
            if len(parts)>=3: W_Float[int(parts[0]), int(parts[1])] = float(parts[2])

    # Output Raw
    with open(FILE_RAW_OUT) as f: vals = [float(x) for x in f.read().split()]
    Y_Gold_Float = np.array(vals).reshape(SEQ_LEN, D_MODEL)

    return X_Hex, W_Hex, Y_Gold_Hex, X_Float, W_Float, Y_Gold_Float

# ==============================================================================
# HÀM PHÂN TÍCH CHI TIẾT
# ==============================================================================
def analyze(t, out_ch, x_hex, w_hex, gold_hex, x_flt, w_flt, gold_flt):
    print(f"\n=======================================================================")
    print(f"!!! DEEP DIVE AT TOKEN {t}, CHANNEL {out_ch} !!!")
    print(f"=======================================================================")
    
    # 1. TÍNH TOÁN FLOAT THUẦN TÚY
    dot_float = np.dot(x_flt, w_flt)
    dot_float_fixed = to_fixed(dot_float) # Convert kết quả float sang Int để so sánh
    
    # 2. TÍNH TOÁN HEX PRECISE (Add then Shift)
    acc_precise = 0
    for i in range(D_INNER):
        acc_precise += int(x_hex[i]) * int(w_hex[i])
    res_precise = sat16(acc_precise >> FRAC_BITS)
    
    # 3. TÍNH TOÁN HARDWARE (Shift then Add)
    acc_hw = 0
    for i in range(D_INNER):
        prod = int(x_hex[i]) * int(w_hex[i])
        acc_hw += (prod >> FRAC_BITS)
    res_hw = sat16(acc_hw)

    # --- IN BẢNG SO SÁNH ---
    print(f"{'Type':<20} | {'Value (Dec)':<15} | {'Value (Hex)':<10} | {'Diff vs Gold Hex'}")
    print("-" * 70)
    print(f"{'Golden File (Hex)':<20} | {gold_hex:<15} | {gold_hex & 0xFFFF:04x}")
    print(f"{'Raw Float (Calc)':<20} | {dot_float:<15.4f} | {'(N/A)'}")
    print(f"{'Raw Float -> Int':<20} | {dot_float_fixed:<15} | {dot_float_fixed & 0xFFFF:04x}     | {abs(dot_float_fixed - gold_hex)}")
    print(f"{'Hex Precise (Sim)':<20} | {res_precise:<15} | {res_precise & 0xFFFF:04x}     | {abs(res_precise - gold_hex)}")
    print(f"{'Hardware (Sim)':<20} | {res_hw:<15} | {res_hw & 0xFFFF:04x}     | {abs(res_hw - gold_hex)}")
    
    print("-" * 70)
    print("PHÂN TÍCH:")
    
    # So sánh Float Calc vs Hex Golden
    if abs(dot_float_fixed - gold_hex) < 5:
        print("1. Toán học Float và Golden Hex KHỚP NHAU -> File Golden chuẩn.")
    else:
        print(f"1. Toán học Float và Golden Hex LỆCH NHAU ({abs(dot_float_fixed - gold_hex)}).")
        print("   -> Có thể do sai số khi convert Input Float -> Input Hex.")

    # So sánh Hex Precise vs Hardware
    loss = abs(res_precise - res_hw)
    print(f"2. Hardware mất {loss} đơn vị so với tính chính xác (do Shift-then-Add).")
    
    # So sánh Hex Precise vs Float Calc
    quant_error = abs(res_precise - dot_float_fixed)
    print(f"3. Hex Input sai lệch so với Float Input gây ra sai số: {quant_error} đơn vị.")

# ==============================================================================
# MAIN
# ==============================================================================
def run():
    X_H, W_H, Y_GH, X_F, W_F, Y_GF = load_all_data()
    
    print(f"Scanning for Diff > {TRIGGER_DIFF}...")
    
    for t in range(SEQ_LEN):
        for out_ch in range(D_MODEL):
            # Tính nhanh HW
            acc = 0
            for k in range(D_INNER):
                acc += (int(X_H[t,k]) * int(W_H[out_ch,k])) >> FRAC_BITS
            val_hw = sat16(acc)
            
            if abs(val_hw - Y_GH[t, out_ch]) > TRIGGER_DIFF:
                analyze(t, out_ch, X_H[t], W_H[out_ch], Y_GH[t, out_ch], 
                        X_F[t], W_F[out_ch], Y_GF[t, out_ch])
                return 

if __name__ == "__main__":
    run()