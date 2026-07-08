import numpy as np
import math

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_MODEL = 64
DATA_WIDTH = 16
FRAC_BITS = 12
MAX_VAL = (2**(DATA_WIDTH-1) - 1)
MIN_VAL = -(2**(DATA_WIDTH-1))
SCALE = 2**FRAC_BITS
EPSILON = 1e-5

# --- HÀM HỖ TRỢ FIXED-POINT ---
def float_to_fixed(f):
    # Mô phỏng bão hòa đầu vào
    val = int(round(f * SCALE))
    if val > MAX_VAL: val = MAX_VAL
    if val < MIN_VAL: val = MIN_VAL
    return val & 0xFFFF # Trả về Hex 16-bit

def fixed_to_float(hex_val):
    if hex_val >= 32768: hex_val -= 65536
    return hex_val / SCALE

def to_hex_str(val):
    return f"{val:04x}"

# --- MÔ PHỎNG PHẦN CỨNG RMSNORM ---
def hardware_rsqrt(variance_fixed):
    # Input là số Fixed Point (Int)
    # Convert ngược về float để tính toán (giả lập ROM)
    var_f = fixed_to_float(variance_fixed)
    
    if var_f <= 0:
        res_f = 0.0
    else:
        res_f = 1.0 / math.sqrt(var_f + EPSILON)
    
    return float_to_fixed(res_f)

def generate_data():
    print("Generating Real Data for RMSNorm...")
    
    # 1. Tạo Input (Random chuẩn Q3.12 range -4 đến 4)
    # Shape: (1000, 64)
    input_data = np.random.uniform(-4.0, 4.0, (SEQ_LEN, D_MODEL))
    
    # 2. Tạo Weight (Gamma) (Random dương từ 0.5 đến 1.5)
    # Shape: (64,) - Dùng chung cho mọi token
    weights = np.random.uniform(0.5, 1.5, (D_MODEL,))
    
    # 3. Tính Golden Output (Mô phỏng logic phần cứng)
    golden_output = np.zeros_like(input_data)
    
    # Mảng để lưu file Hex
    hex_input = []
    hex_weight = []
    hex_output = []
    
    # -- Xử lý Weight trước --
    for w in weights:
        hex_weight.append(to_hex_str(float_to_fixed(w)))
        
    # -- Xử lý từng Token --
    for t in range(SEQ_LEN):
        x_vec = input_data[t]
        
        # A. Tính Tổng Bình Phương (Fixed Point Acc)
        sum_sq = 0
        x_fixed_list = []
        
        for val in x_vec:
            fx = float_to_fixed(val)
            x_fixed_list.append(to_hex_str(fx))
            
            # Mô phỏng PE: x * x (Q3.12 * Q3.12 -> Q6.24 -> Shift 12 -> Q3.12)
            # Lưu ý: Verilog logic của cậu: sum += x*x (đã shift)
            # Ta cần convert fx sang signed int python để tính
            sx = fx if fx < 32768 else fx - 65536
            sq = (sx * sx) >> FRAC_BITS 
            # --- THÊM ĐOẠN NÀY ĐỂ GIỐNG HARDWARE ---
            # Mô phỏng bão hòa đầu ra của PE (16-bit Q3.12)
            #if sq > MAX_VAL: sq = MAX_VAL
            #if sq < MIN_VAL: sq = MIN_VAL # (Thực ra sq luôn dương nên ko cần dòng này, nhưng cứ để cho chắc)
            # ---------------------------------------
            sum_sq += sq
            
        hex_input.extend(x_fixed_list)
        
        # B. Tính Mean (Shift >> 6)
        mean_sq = sum_sq >> 6
        # Kẹp 16 bit
        if mean_sq > MAX_VAL: mean_sq = MAX_VAL
        
        # C. Tra RSqrt (Mô phỏng ROM)
        # Lưu ý: mean_sq đang là int, cần & 0xFFFF để giả lập input 16bit unsigned cho ROM
        rsqrt_val = hardware_rsqrt(mean_sq & 0xFFFF)
        
        # Convert rsqrt_val về signed int để nhân
        s_rsqrt = rsqrt_val if rsqrt_val < 32768 else rsqrt_val - 65536
        
        # D. Nhân Output: y = x * w * S
        # Hardware: (x * w) >> 12, sau đó * S >> 12
        for i in range(D_MODEL):
            # Input x
            sx = float_to_fixed(x_vec[i])
            if sx >= 32768: sx -= 65536
            
            # Weight w
            sw = float_to_fixed(weights[i])
            if sw >= 32768: sw -= 65536
            
            # Pass 1: x * w
            pass1 = (sx * sw) >> FRAC_BITS
            
            # Pass 2: pass1 * S
            pass2 = (pass1 * s_rsqrt) >> FRAC_BITS
            
            # Saturation
            if pass2 > MAX_VAL: pass2 = MAX_VAL
            if pass2 < MIN_VAL: pass2 = MIN_VAL
            
            hex_output.append(to_hex_str(pass2 & 0xFFFF))

    # 4. Ghi File
    with open("rms_real_input.txt", "w") as f:
        f.write("\n".join(hex_input))
    
    with open("rms_real_weight.txt", "w") as f:
        f.write("\n".join(hex_weight))
        
    with open("rms_real_golden.txt", "w") as f:
        f.write("\n".join(hex_output))
        
    print("DONE! Files generated:")
    print("- rms_real_input.txt (64000 lines)")
    print("- rms_real_weight.txt (64 lines)")
    print("- rms_real_golden.txt (64000 lines)")

if __name__ == "__main__":
    generate_data()