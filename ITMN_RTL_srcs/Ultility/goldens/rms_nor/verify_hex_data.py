import math

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_MODEL = 64
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 2**FRAC_BITS
MAX_VAL = 32767
MIN_VAL = -32768
EPSILON = 1e-5

# Tên file Hex (Đã convert từ C++)
FILE_INPUT  = "rms_ptb_input.txt"
FILE_WEIGHT = "rms_ptb_weight.txt"
FILE_GOLDEN = "rms_ptb_golden.txt"

# --- HÀM HỖ TRỢ ---
def hex_to_int(hex_str):
    val = int(hex_str, 16)
    if val >= 32768: val -= 65536
    return val

def int_to_hex(val):
    return f"{val & 0xFFFF:04x}"

def sat(val):
    if val > MAX_VAL: return MAX_VAL
    if val < MIN_VAL: return MIN_VAL
    return val

# Giả lập RSqrt ROM (Dùng float để tính chuẩn giá trị ROM)
def get_rsqrt_rom(int_val):
    # int_val là Q3.12 unsigned (Mean bình phương)
    float_val = int_val / SCALE
    if float_val <= 0: return 0
    res = 1.0 / math.sqrt(float_val + EPSILON)
    return sat(int(round(res * SCALE)))

def verify():
    print("Loading files...")
    try:
        with open(FILE_INPUT, 'r') as f: inputs = [hex_to_int(line.strip()) for line in f]
        with open(FILE_WEIGHT, 'r') as f: weights = [hex_to_int(line.strip()) for line in f]
        with open(FILE_GOLDEN, 'r') as f: goldens = [hex_to_int(line.strip()) for line in f]
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file .txt. Hãy chạy convert_ptb_data.py trước.")
        return

    print(f"Loaded: In={len(inputs)}, W={len(weights)}, Gold={len(goldens)}")
    
    total_errors = 0
    max_diff = 0
    
    # --- VÒNG LẶP MÔ PHỎNG HARDWARE ---
    for t in range(SEQ_LEN):
        # 1. Lấy dữ liệu của 1 token
        start_idx = t * D_MODEL
        x_vec = inputs[start_idx : start_idx + D_MODEL]
        gold_vec = goldens[start_idx : start_idx + D_MODEL]
        
        # 2. Bước 1: Tính Tổng Bình Phương (Có Shift >> 2 như Hardware)
        sum_sq = 0
        for x in x_vec:
            # Hardware: pe_in = x >>> 2
            x_sh = x >> 2 
            # PE Mode MAC: (A * B) >> 12
            sq = (x_sh * x_sh) >> FRAC_BITS
            sum_sq += sq
            
        # 3. Bước 2: Tính Mean & RSqrt
        # Hardware: Mean = Sum >> 6
        mean = sum_sq >> 6
        
        # Tra ROM
        S = get_rsqrt_rom(mean)
        
        # Hardware: S = S >>> 2 (Bù lại việc chia input)
        S_adjusted = S >> 2
        
        # 4. Bước 3: Nhân Output (Pass 1 & Pass 2)
        for i in range(D_MODEL):
            x = x_vec[i]
            w = weights[i] # Weight dùng chung
            
            # Pass 1: Tmp = x * w
            # Hardware PE: (x * w) >> 12
            pass1 = (x * w) >> FRAC_BITS
            
            # Pass 2: Out = Tmp * S_adjusted
            # Hardware PE: (pass1 * S) >> 12
            out_calc = (pass1 * S_adjusted) >> FRAC_BITS
            
            # Saturation output
            out_final = sat(out_calc)
            
            # 5. So sánh với Golden
            gold = gold_vec[i]
            diff = abs(out_final - gold)
            
            if diff > max_diff: max_diff = diff
            
            # Ngưỡng chấp nhận (ví dụ 10 đơn vị)
            if diff > 10:
                if total_errors < 10: # Chỉ in 10 lỗi đầu
                    print(f"[ERR] T={t}, i={i} | In={x}, W={w} | Calc={out_final} ({int_to_hex(out_final)}), Gold={gold} ({int_to_hex(gold)}) | Diff={diff}")
                    # Debug chi tiết cho lỗi đầu tiên
                    if total_errors == 0:
                        print(f"   -> Debug: SumSq={sum_sq}, Mean={mean}, ROM_S={S}, S_adj={S_adjusted}")
                        print(f"   -> Pass1(x*w)={pass1}, Pass2(p1*S)={out_calc}")
                total_errors += 1

    print("\n" + "="*30)
    print(f"VERIFICATION DONE.")
    print(f"Total Errors (>10 LSB): {total_errors}")
    print(f"Max Difference: {max_diff}")
    
    if total_errors == 0:
        print(">> PYTHON CONFIRMS: Hardware Logic matches Data!")
    else:
        print(">> MISMATCH: Hardware Logic (Shift trick) does NOT match the C++ Golden Data.")
        print("   Reason: C++ likely uses Float32 or pure Fixed-point without shift-trick.")

if __name__ == "__main__":
    verify()