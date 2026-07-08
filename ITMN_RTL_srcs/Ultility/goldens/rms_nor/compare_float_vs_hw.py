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

# File Hex đầu vào (Phải đảm bảo đúng file cậu đang dùng cho TB)
FILE_INPUT  = "rms_ptb_input.txt"
FILE_WEIGHT = "rms_ptb_weight.txt"

# FILE_INPUT  = "rms_real_input.txt"
# FILE_WEIGHT = "rms_real_weight.txt"

# --- HÀM HỖ TRỢ ---
def hex_to_float(hex_str):
    val = int(hex_str, 16)
    if val >= 32768: val -= 65536
    return val / SCALE

def float_to_hex_int(f_val):
    val = int(round(f_val * SCALE))
    if val > MAX_VAL: val = MAX_VAL
    if val < MIN_VAL: val = MIN_VAL
    return val

def int_to_hex_str(val):
    return f"{val & 0xFFFF:04x}"

def sat(val):
    if val > MAX_VAL: return MAX_VAL
    if val < MIN_VAL: return MIN_VAL
    return val

# Giả lập RSqrt ROM cho Hardware Model
def get_rsqrt_rom(int_val):
    float_val = int_val / SCALE
    if float_val <= 0: return 0
    res = 1.0 / math.sqrt(float_val + EPSILON)
    return sat(int(round(res * SCALE)))

def run_comparison():
    print("Loading Input/Weight files...")
    try:
        with open(FILE_INPUT, 'r') as f: 
            inputs_hex = [line.strip() for line in f]
        with open(FILE_WEIGHT, 'r') as f: 
            weights_hex = [line.strip() for line in f]
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file input/weight.")
        return

    # Convert toàn bộ sang Float để tính toán chuẩn Toán học
    inputs_f = [hex_to_float(h) for h in inputs_hex]
    weights_f = [hex_to_float(h) for h in weights_hex]
    
    # Mảng lưu kết quả để ghi file
    float_golden_hex_list = []
    
    total_diff = 0
    max_diff = 0
    err_count = 0
    
    print("\n--- STARTING COMPARISON ---")
    print(f"Checking {SEQ_LEN} tokens...")

    for t in range(SEQ_LEN):
        start = t * D_MODEL
        end = start + D_MODEL
        
        # Lấy vector float cho token t
        x_vec_f = inputs_f[start:end]
        # Lấy vector int cho Hardware Model
        x_vec_int = [int(inputs_hex[i], 16) for i in range(start, end)]
        # Xử lý số âm cho int
        x_vec_int = [x if x < 32768 else x - 65536 for x in x_vec_int]
        
        weight_vec_f = weights_f # Weight dùng chung (64 phần tử)
        weight_vec_int = [int(w, 16) for w in weights_hex]
        weight_vec_int = [w if w < 32768 else w - 65536 for w in weight_vec_int]

        # ---------------------------------------------------------
        # 1. TÍNH TOÁN KIỂU FLOAT (Toán học thuần túy - IDEAL)
        # ---------------------------------------------------------
        # Sum Square
        sum_sq_f = sum([x**2 for x in x_vec_f])
        # Mean
        mean_f = sum_sq_f / D_MODEL
        # RSqrt
        rsqrt_f = 1.0 / math.sqrt(mean_f + EPSILON)
        
        # Output Float
        out_f_vec = []
        for i in range(D_MODEL):
            val = x_vec_f[i] * weight_vec_f[i] * rsqrt_f
            out_f_vec.append(val)
            # Lưu lại dạng Hex để dùng làm Golden mới
            float_golden_hex_list.append(int_to_hex_str(float_to_hex_int(val)))

        # ---------------------------------------------------------
        # 2. TÍNH TOÁN KIỂU HARDWARE (Q3.12 + Shift Trick)
        # ---------------------------------------------------------
        sum_sq_hw = 0
        for x in x_vec_int:
            x_sh = x >> 2 # Shift trick
            sq = (x_sh * x_sh) >> FRAC_BITS
            if sq > MAX_VAL: sq = MAX_VAL # Saturation
            sum_sq_hw += sq
            
        mean_hw = sum_sq_hw >> 6
        S_rom = get_rsqrt_rom(mean_hw)
        S_adj = S_rom >> 1 # Bù lại shift
        
        out_hw_vec = []
        for i in range(D_MODEL):
            x = x_vec_int[i]
            w = weight_vec_int[i]
            
            # Pass 1
            pass1 = (x * w) >> FRAC_BITS
            # Pass 2
            pass2 = (pass1 * S_adj) >> FRAC_BITS
            # Saturation
            out_hw = sat(pass2)
            out_hw_vec.append(out_hw)

        # ---------------------------------------------------------
        # 3. SO SÁNH (Float Ideal vs Hardware Model)
        # ---------------------------------------------------------
        for i in range(D_MODEL):
            # Convert Float Ideal sang Int Q3.12 để so sánh
            gold_int = float_to_hex_int(out_f_vec[i])
            hw_int = out_hw_vec[i]
            
            diff = abs(gold_int - hw_int)
            if diff > max_diff: max_diff = diff
            
            # Nếu lệch quá 15 đơn vị (~0.3%) thì báo lỗi
            if diff > 15: 
                err_count += 1
                if err_count <= 5: # Chỉ in 5 lỗi đầu
                    print(f"[Diff] T={t} i={i} | Float={gold_int} vs HW={hw_int} | Diff={diff}")
                    if err_count == 1:
                        print(f"   -> Debug Float: Mean={mean_f:.4f}, RSqrt={rsqrt_f:.4f}")
                        print(f"   -> Debug HW   : Mean={mean_hw} (raw), RSqrt_Rom={S_rom}, S_adj={S_adj}")

    # Ghi file Golden mới (Dựa trên tính toán Float chuẩn)
    with open("rms_computed_float_golden.txt", "w") as f:
        f.write("\n".join(float_golden_hex_list))

    print("-" * 30)
    print(f"Max Difference: {max_diff} LSB")
    print(f"Total Significant Errors (>15 LSB): {err_count}")
    
    if err_count < 100: # Chấp nhận một số lượng nhỏ sai số biên
        print("\n>>> KẾT LUẬN: Logic Hardware (Shift Trick) CHẤP NHẬN ĐƯỢC <<<")
        print("Hãy dùng file 'rms_computed_float_golden.txt' làm Golden cho Testbench.")
    else:
        print("\n>>> KẾT LUẬN: Sai số quá lớn! Cần xem lại chiến thuật Shift.")

if __name__ == "__main__":
    run_comparison()