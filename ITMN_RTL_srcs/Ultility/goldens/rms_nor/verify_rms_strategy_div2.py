import math

# --- CẤU HÌNH HỆ THỐNG ---
SEQ_LEN = 1000
D_MODEL = 64
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 2**FRAC_BITS
MAX_VAL = 32767
MIN_VAL = -32768
EPSILON = 1e-5

# Tên file (Đảm bảo đúng tên file cậu đang có)
FILE_INPUT  = "rms_ptb_input.txt"
FILE_WEIGHT = "rms_ptb_weight.txt"
FILE_GOLDEN_CPP = "rms_ptb_golden.txt" # File gốc từ C++ (Float chuẩn)
FILE_OUTPUT_HW  = "rms_hardware_golden.txt" # File sẽ tạo ra cho Verilog

# --- HÀM HỖ TRỢ FIXED-POINT ---
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

# --- GIẢ LẬP ROM RSQRT (Chiến thuật CHIA 2) ---
# Input: Mean_HW (đã bị chia 16 do Input >> 2)
# Output: RSqrt_Real / 2 (Để nhét vừa Q3.12)
def get_rsqrt_rom_simulated(mean_hw_int):
    # 1. Convert Mean HW sang Float
    mean_hw_float = mean_hw_int / SCALE
    
    # 2. Khôi phục Mean thực tế (Nhân 16 bù lại việc Shift input)
    mean_real_float = mean_hw_float * 16.0
    
    if mean_real_float <= 0: return 0
    
    # 3. Tính RSqrt thực tế
    rsqrt_real = 1.0 / math.sqrt(mean_real_float + EPSILON)
    
    # 4. Chia 2 để nhét vừa ROM Q3.12 (Scale Factor = 0.5)
    rsqrt_rom_target = rsqrt_real * 0.5
    
    # 5. Convert về Fixed Point
    return sat(int(round(rsqrt_rom_target * SCALE)))

def run_verification():
    print("Loading files...")
    try:
        with open(FILE_INPUT, 'r') as f: inputs = [hex_to_int(line.strip()) for line in f]
        with open(FILE_WEIGHT, 'r') as f: weights = [hex_to_int(line.strip()) for line in f]
        with open(FILE_GOLDEN_CPP, 'r') as f: goldens_cpp = [hex_to_int(line.strip()) for line in f]
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file input/weight/golden.")
        return

    print(f"Loaded Data. Starts computing with Strategy: Input>>2, ROM/2, Out<<1")
    
    hw_results = []
    total_diff = 0
    max_diff = 0
    err_count = 0
    
    # --- VÒNG LẶP TÍNH TOÁN ---
    for t in range(SEQ_LEN):
        # Cắt dữ liệu cho 1 token
        start = t * D_MODEL
        end = start + D_MODEL
        x_vec = inputs[start:end]
        
        # 1. Tính Tổng Bình Phương (Input >> 2)
        sum_sq_hw = 0
        for x in x_vec:
            x_sh = x >> 2 # Shift trick
            # PE MAC: (A * B) >> 12
            sq = (x_sh * x_sh) >> FRAC_BITS
            sum_sq_hw += sq # Cộng dồn (32-bit trong Verilog, Python tự lo)
            
        # 2. Tính Mean & Tra ROM
        mean_hw = sum_sq_hw >> 6
        S_rom = get_rsqrt_rom_simulated(mean_hw) 
        
        # 3. Nhân Output
        for i in range(D_MODEL):
            x = x_vec[i]
            w = weights[i] # Weight chung cho 64 kênh
            
            # Pass 1: x * w
            pass1 = (x * w) >> FRAC_BITS
            
            # Pass 2: Pass1 * S_rom
            pass2 = (pass1 * S_rom) >> FRAC_BITS
            
            # Final Shift: Dịch trái 1 để bù lại việc ROM chia 2
            # Lưu ý: Cần bão hòa sau khi dịch
            final_val = pass2 << 1
            final_val = sat(final_val)
            
            hw_results.append(final_val)
            
            # 4. So sánh với C++ Golden
            gold = goldens_cpp[start + i]
            diff = abs(final_val - gold)
            
            if diff > max_diff: max_diff = diff
            
            # Ngưỡng cảnh báo sai số (ví dụ 20 LSB ~ 0.5%)
            if diff > 20:
                err_count += 1
                if err_count <= 5: # In mẫu vài lỗi
                    print(f"[Diff] T={t} i={i} | C++={gold} ({int_to_hex(gold)}) vs HW={final_val} ({int_to_hex(final_val)}) | Diff={diff}")

    # --- KẾT QUẢ ---
    print("-" * 40)
    print(f"Max Difference vs C++ Float: {max_diff} LSB")
    print(f"Total Large Errors (>20 LSB): {err_count} / {len(hw_results)}")
    
    # Xuất file Golden mới cho Verilog
    with open(FILE_OUTPUT_HW, "w") as f:
        for val in hw_results:
            f.write(f"{int_to_hex(val)}\n")
            
    print(f"\n>> Đã tạo file '{FILE_OUTPUT_HW}'.")
    print(">> Hãy dùng file này nạp vào Testbench Verilog (mem_golden).")
    print(">> Nếu Verilog chạy khớp với file này -> Phần cứng ĐÚNG THIẾT KẾ.")

if __name__ == "__main__":
    run_verification()