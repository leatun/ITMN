import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_MODEL = 64
FRAC_BITS = 11
SCALE = 1 << FRAC_BITS

# ĐIỂM SOI (Lấy từ kết quả lần trước)
TARGET_TOK = 974
TARGET_CH  = 0

# --- FILES HEX ---
FILE_HEX_IN     = "debug_input_correct.txt"
FILE_HEX_W      = "debug_weight_correct.txt"

# --- FILES RAW FLOAT ---
FILE_RAW_IN     = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_RAW_W      = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"

# ==============================================================================
# HÀM LOAD DATA
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str.strip(), 16)
    return val - 65536 if val & 0x8000 else val

def float_to_fixed(val):
    return int(round(val * SCALE))

def load_data():
    print("--- Loading Data ---")
    
    # 1. Hex Data
    with open(FILE_HEX_IN) as f: 
        X_Hex = np.array([to_signed(l) for l in f]).reshape(SEQ_LEN, D_INNER)
    
    with open(FILE_HEX_W) as f: w_flat = [to_signed(l) for l in f]
    W_Hex = np.zeros((D_MODEL, D_INNER), dtype=int)
    idx = 0
    for chunk in range(4):
        for col in range(128):
            for r in range(16):
                if idx < len(w_flat): W_Hex[chunk*16+r, col] = w_flat[idx]; idx+=1

    # 2. Raw Float Data
    with open(FILE_RAW_IN) as f: vals = [float(x) for x in f.read().split()]
    X_Raw = np.array(vals).reshape(D_INNER, SEQ_LEN).T # Transpose (1000, 128)
    
    W_Raw = np.zeros((D_MODEL, D_INNER))
    with open(FILE_RAW_W) as f:
        for line in f:
            parts = line.replace('[','').replace(']','').replace(',',' ').split()
            if len(parts)>=3: W_Raw[int(parts[0]), int(parts[1])] = float(parts[2])

    return X_Hex, W_Hex, X_Raw, W_Raw

# ==============================================================================
# HÀM PHÂN TÍCH
# ==============================================================================
def analyze_step_by_step():
    X_H, W_H, X_F, W_F = load_data()
    
    print(f"\n=====================================================================")
    print(f"!!! BREAKDOWN ANALYSIS: TOKEN {TARGET_TOK}, CHANNEL {TARGET_CH} !!!")
    print(f"=====================================================================")
    
    # Lấy Vector tại điểm cần soi
    vec_x_hex = X_H[TARGET_TOK]
    vec_w_hex = W_H[TARGET_CH]
    vec_x_raw = X_F[TARGET_TOK]
    vec_w_raw = W_F[TARGET_CH]
    
    print(f"{'i':<3} | {'X Raw':<10} {'X Hex':<6} {'Diff':<4} | {'W Raw':<10} {'W Hex':<6} {'Diff':<4} || {'Prod Raw':<12} {'Prod Hex':<10} | {'Acc Err':<8}")
    print("-" * 110)
    
    sum_raw = 0.0
    sum_hex_precise = 0 # Accumulate 32-bit (chưa shift)
    
    for i in range(D_INNER):
        # 1. Input Analysis
        x_f = vec_x_raw[i]
        x_h = vec_x_hex[i]
        x_expect = float_to_fixed(x_f)
        x_diff = x_h - x_expect # Nếu khác 0 nghĩa là file hex nạp vào khác với file raw
        
        # 2. Weight Analysis
        w_f = vec_w_raw[i]
        w_h = vec_w_hex[i]
        w_expect = float_to_fixed(w_f)
        w_diff = w_h - w_expect
        
        # 3. Product Analysis
        prod_f = x_f * w_f
        prod_h = int(x_h) * int(w_h) # Q6.24
        
        # Convert Prod Hex về Float Scale để so sánh
        prod_h_scaled = prod_h / (SCALE * SCALE) 
        
        sum_raw += prod_f
        sum_hex_precise += prod_h
        
        # Sai số tích lũy tại bước này (đã scale về Int 16-bit để dễ hình dung)
        # Acc_Error = (Sum_Hex_Precise >> 12) - to_fixed(Sum_Raw)
        curr_hex_val = sum_hex_precise >> FRAC_BITS
        curr_raw_val = float_to_fixed(sum_raw)
        acc_error = curr_hex_val - curr_raw_val
        
        # In các dòng có sai lệch input hoặc sai số tích lũy lớn, và vài dòng đầu
        if i < 10 or abs(x_diff) > 0 or abs(w_diff) > 0 or i > 120:
            print(f"{i:<3} | {x_f:<10.4f} {x_h:<6} {x_diff:<4} | {w_f:<10.4f} {w_h:<6} {w_diff:<4} || {prod_f:<12.4f} {prod_h_scaled:<10.4f} | {acc_error:<8}")

    print("-" * 110)
    
    # KẾT QUẢ CUỐI CÙNG
    print("FINAL RESULT:")
    final_raw_int = float_to_fixed(sum_raw)
    final_hex_int = sum_hex_precise >> FRAC_BITS
    
    print(f"Raw Float Sum   : {sum_raw:.6f} -> Convert to Int16: {final_raw_int}")
    print(f"Hex Precise Sum : {final_hex_int} (Logic: Sum(Hex*Hex) >> 12)")
    print(f"DIFFERENCE      : {final_hex_int - final_raw_int}")
    
    print("\nKẾT LUẬN:")
    if final_hex_int - final_raw_int == 970: # Khớp với số 970 lúc nãy
        print("-> Sai số 970 ĐƯỢC TẠO RA DO TÍCH LŨY CÁC SAI SỐ LÀM TRÒN NHỎ (Quantization Noise).")
        print("-> Mỗi phép nhân (X_tròn * W_tròn) lệch một chút so với (X*W). Cộng 128 lần lại thành lệch lớn.")
    else:
        print("-> Có vấn đề khác (ví dụ: X_Hex hoặc W_Hex nạp vào bị sai). Kiểm tra cột Diff của X và W ở bảng trên.")

if __name__ == "__main__":
    analyze_step_by_step()