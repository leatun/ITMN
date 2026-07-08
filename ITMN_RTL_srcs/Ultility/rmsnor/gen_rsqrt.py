import numpy as np
import math

# --- Cấu hình Fixed Point Q3.12 ---
DATA_WIDTH = 16
FRAC_BITS = 12
INT_BITS = 3  # Không tính bit dấu
# Max value ≈ 7.9997, Min value = -8.0
MAX_VAL = (2**(DATA_WIDTH-1) - 1) / 2**FRAC_BITS
MIN_VAL = -(2**(DATA_WIDTH-1)) / 2**FRAC_BITS

EPSILON = 1e-5  # Để tránh chia cho 0 (chuẩn RMSNorm)

def to_fixed(f_val):
    # Kẹp giá trị (Saturate)
    if f_val > MAX_VAL:
        f_val = MAX_VAL
    elif f_val < MIN_VAL:
        f_val = MIN_VAL
    
    # Nhân với 2^Frac
    scaled = int(round(f_val * (2**FRAC_BITS)))
    
    # Xử lý số âm (Two's Complement)
    if scaled < 0:
        scaled = (1 << DATA_WIDTH) + scaled
        
    return scaled & 0xFFFF

def from_fixed(hex_val):
    # Xử lý số âm
    if hex_val >= (1 << (DATA_WIDTH - 1)):
        hex_val -= (1 << DATA_WIDTH)
    return hex_val / (2**FRAC_BITS)

def generate_rsqrt_mem():
    filename = "rsqrt_rom.mem"
    print(f"Generatring {filename} for Q3.{FRAC_BITS}...")
    
    with open(filename, 'w') as f:
        # Quét hết không gian địa chỉ 16-bit (0 -> 65535)
        # Địa chỉ chính là bit pattern của input Q3.12
        for i in range(65536):
            # 1. Giải mã Input từ Hex sang Float
            input_float = from_fixed(i)
            
            # 2. Tính RSqrt
            # RMSNorm chỉ quan tâm số dương (Variance >= 0)
            # Nếu input <= 0 (do nhiễu hoặc chưa khởi tạo), ta gán output = 0 hoặc Max tùy ý
            # Ở đây tôi gán = 0 cho an toàn.
            
            if input_float <= 0:
                # Thực tế Variance luôn >= 0, nhưng nếu input âm thì trả về 0
                # Lưu ý: input_float = 0 cũng sẽ nhảy vào đây
                result_float = 0.0
            else:
                # Công thức: 1 / sqrt(x + epsilon)
                # Epsilon giúp tránh lỗi khi x rất nhỏ gần 0
                val_with_eps = input_float + EPSILON
                result_float = 1.0 / math.sqrt(val_with_eps)
            
            # 3. Chuyển kết quả về Fixed Point Hex
            res_fixed = to_fixed(result_float)
            
            # 4. Ghi vào file (Format Hex 4 ký tự)
            f.write(f"{res_fixed:04x}\n")
            
    print(f"Done! Saved to {filename}")
    print(f"Test values:")
    print(f"  Input 1.0 (Hex 1000) -> Output {to_fixed(1.0/math.sqrt(1.0)):04x}")
    print(f"  Input 4.0 (Hex 4000) -> Output {to_fixed(1.0/math.sqrt(4.0)):04x} (Should be 0.5 -> 0800)")
    print(f"  Input 0.01 (Hex 0029) -> Output {to_fixed(1.0/math.sqrt(0.01)):04x} (Should saturate to MAX)")

if __name__ == "__main__":
    generate_rsqrt_mem()