import numpy as np
import math

DATA_WIDTH = 16
FRAC_BITS = 12
MAX_VAL = 7.999
MIN_VAL = -8.0
SCALE = 2**FRAC_BITS
EPSILON = 1e-5

# --- CHIẾN THUẬT MỚI: ROM SCALING ---
# Vì ta chia input cho 4 (Shift >> 2) để tránh tràn bình phương
# -> Mean bị chia 16 -> RSqrt thực tế sẽ NHÂN 4.
# -> Cộng thêm việc RSqrt gốc đã lớn (9.6).
# -> Tổng cộng giá trị cần lưu là rất lớn.
# TA SẼ LƯU GIÁ TRỊ: (RSqrt_Calculated / 8)
# Tại sao /8? 
# - Chia 4 để bù lại việc Input Shift.
# - Chia thêm 2 nữa để nén giá trị to (9.6) xuống vùng an toàn (4.82).
ROM_SCALE_FACTOR = 1.0 / 8.0 

def to_fixed(f_val):
    scaled = int(round(f_val * SCALE))
    if scaled > 32767: scaled = 32767
    if scaled < -32768: scaled = -32768
    if scaled < 0: scaled = 65536 + scaled
    return scaled & 0xFFFF

def generate_rsqrt_mem():
    filename = "rsqrt_rom.mem"
    print(f"Generating Scaled {filename}...")
    
    with open(filename, 'w') as f:
        for i in range(65536):
            # i là Input (Mean) dạng Q3.12 unsigned
            input_float = i / SCALE
            
            if input_float <= 0:
                result_float = 0.0
            else:
                # Tính RSqrt chuẩn
                raw_rsqrt = 1.0 / math.sqrt(input_float + EPSILON)
                # THU NHỎ GIÁ TRỊ ĐỂ NHÉT VỪA ROM
                result_float = raw_rsqrt * ROM_SCALE_FACTOR
            
            f.write(f"{to_fixed(result_float):04x}\n")
            
    print("Done! ROM data is now scaled down by 1/8.")

if __name__ == "__main__":
    generate_rsqrt_mem()