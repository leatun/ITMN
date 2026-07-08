import numpy as np

# --- CẤU HÌNH ---
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
NUM_SEGMENTS = 64  # Chia thành 64 đoạn
# Input Q3.12 range: -8.0 to +7.999
# Ta dùng 6 bit cao nhất (MSB) của input để làm địa chỉ Segment
# Bit [15:10] -> 64 giá trị

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def to_hex(val, width=16):
    val = int(val)
    if val < 0: val = (1 << width) + val
    return f"{val & ((1<<width)-1):04x}"

def float_to_fixed(val):
    return int(round(val * SCALE))

print("Generating SiLU PWL Coefficients (64 Segments)...")

slope_list = []
intercept_list = []

# Dải giá trị của Q3.12 là khoảng -8 đến 8
# Mỗi segment rộng = 16.0 / 64 = 0.25
step = 0.25

with open("silu_pwl_coeffs.mem", "w") as f:
    for i in range(NUM_SEGMENTS):
        # Xác định khoảng giá trị của segment i
        # i chạy từ 0 đến 63.
        # Với số bù 2, index 0 tương ứng với 0.0
        # Index 32 tương ứng với -8.0 (bit sign = 1)
        # Ta cần map lại index của vòng lặp i sang giá trị thực tế
        
        # Logic phần cứng: Address = in_data[15:10]
        # Nếu in_data = 0x0000 (0.0) -> Addr = 0
        # Nếu in_data = 0x8000 (-8.0) -> Addr = 32
        
        # Convert index i sang giá trị fixed point start
        # i là 6 bit unsigned. Nếu i >= 32, nó đại diện số âm.
        if i < 32:
            start_val = i * step
        else:
            start_val = (i - 64) * step
            
        end_val = start_val + step
        
        # Lấy mẫu trong khoảng này để tìm đường thẳng tốt nhất (Linear Regression)
        x_points = np.linspace(start_val, end_val, 20)
        y_points = silu(x_points)
        
        # Fit y = ax + b
        # a = slope, b = intercept
        slope, intercept = np.polyfit(x_points, y_points, 1)
        
        # Convert sang Fixed Point Q3.12
        a_fixed = float_to_fixed(slope)
        b_fixed = float_to_fixed(intercept)
        
        # Kẹp giá trị (Saturation) cho an toàn
        a_fixed = max(min(a_fixed, 32767), -32768)
        b_fixed = max(min(b_fixed, 32767), -32768)
        
        slope_list.append(a_fixed)
        intercept_list.append(b_fixed)
        
        # Ghi vào file: SLOPE (16 bit) + INTERCEPT (16 bit) = 32 bit line
        # Hardware sẽ đọc 32 bit này và tách ra
        line = f"{to_hex(a_fixed)}{to_hex(b_fixed)}"
        f.write(line + "\n")
        
        # Debug in ra vài dòng
        if i == 0: print(f"Seg 0 (0.0 to 0.25): y = {slope:.3f}x + {intercept:.3f}")
        if i == 32: print(f"Seg 32 (-8.0 to -7.75): y = {slope:.3f}x + {intercept:.3f}")

print("Done! Saved to silu_pwl_coeffs.mem")