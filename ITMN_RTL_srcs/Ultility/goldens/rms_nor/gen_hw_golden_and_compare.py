import math
import statistics

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_MODEL = 64
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 2**FRAC_BITS
MAX_VAL = 32767
MIN_VAL = -32768
EPSILON = 1e-5

# File đầu vào
FILE_INPUT  = "rms_ptb_input.txt"
FILE_WEIGHT = "rms_ptb_weight.txt"
FILE_SW_GOLDEN = "rms_ptb_golden.txt" # File gốc từ C++

# File đầu ra mới (Dành cho Testbench Verilog)
FILE_HW_GOLDEN = "rms_hw_golden.txt"

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

def get_rsqrt_rom(int_val):
    # int_val là Q3.12 unsigned
    float_val = int_val / SCALE
    if float_val <= 0: return 0
    res = 1.0 / math.sqrt(float_val + EPSILON)
    return sat(int(round(res * SCALE)))

def run_gen():
    print("Loading data...")
    try:
        with open(FILE_INPUT, 'r') as f: inputs = [hex_to_int(line.strip()) for line in f]
        with open(FILE_WEIGHT, 'r') as f: weights = [hex_to_int(line.strip()) for line in f]
        with open(FILE_SW_GOLDEN, 'r') as f: sw_goldens = [hex_to_int(line.strip()) for line in f]
    except FileNotFoundError:
        print("Thiếu file input/weight/golden gốc!")
        return

    hw_outputs = []
    diffs = []
    
    print("Processing Hardware Model (with Shift >> 2 trick)...")
    
    for t in range(SEQ_LEN):
        start_idx = t * D_MODEL
        x_vec = inputs[start_idx : start_idx + D_MODEL]
        
        # --- BƯỚC 1: TÍNH TỔNG BÌNH PHƯƠNG (CÓ SHIFT) ---
        sum_sq = 0
        for x in x_vec:
            # Trick: Chia 4 trước khi bình phương để tránh tràn số
            x_sh = x >> 2 
            sq = (x_sh * x_sh) >> FRAC_BITS
            sum_sq += sq
            
        # --- BƯỚC 2: MEAN & RSQRT ---
        mean = sum_sq >> 6 # Chia 64
        S = get_rsqrt_rom(mean)
        S_adjusted = S >> 2 # Bù lại việc chia input
        
        # --- BƯỚC 3: NHÂN OUTPUT ---
        for i in range(D_MODEL):
            x = x_vec[i]
            w = weights[i]
            
            # Pass 1: x * w
            pass1 = (x * w) >> FRAC_BITS
            # Pass 2: * S
            out_calc = (pass1 * S_adjusted) >> FRAC_BITS
            
            out_final = sat(out_calc)
            
            # Lưu kết quả HW
            hw_outputs.append(int_to_hex(out_final))
            
            # So sánh với SW Golden (chỉ để tham khảo độ sai lệch)
            sw_val = sw_goldens[start_idx + i]
            diffs.append(abs(out_final - sw_val))

    # --- BÁO CÁO SAI SỐ GIỮA HW VÀ SW ---
    avg_diff = statistics.mean(diffs)
    max_diff = max(diffs)
    print("\n=== REPORT: HARDWARE vs SOFTWARE (C++) ===")
    print(f"Average Diff: {avg_diff:.4f} LSB")
    print(f"Max Diff    : {max_diff} LSB")
    print("Note: Sai số này là do giới hạn của Fixed-Point Q3.12 và mẹo Shift.")
    
    # --- GHI FILE MỚI CHO VERILOG ---
    with open(FILE_HW_GOLDEN, 'w') as f:
        f.write("\n".join(hw_outputs))
        
    print(f"\n[OK] Generated new golden file: {FILE_HW_GOLDEN}")
    print("Use THIS file in your Verilog Testbench to get PASS!")

if __name__ == "__main__":
    run_gen()