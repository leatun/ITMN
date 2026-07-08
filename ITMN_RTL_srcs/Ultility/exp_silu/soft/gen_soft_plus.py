import numpy as np

# --- CẤU HÌNH ---
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
NUM_SEGMENTS = 64  # 64 đoạn
# Input Q3.12 range: -8.0 to +7.999

def softplus(x):
    # Dùng numpy log và exp để tính toán chính xác
    # np.logaddexp(0, x) tương đương log(1 + exp(x)) nhưng ổn định hơn
    return np.logaddexp(0, x)

def to_hex(val, width=16):
    val = int(val)
    if val < 0: val = (1 << width) + val
    return f"{val & ((1<<width)-1):04x}"

def float_to_fixed(val):
    return int(round(val * SCALE))

print("Generating Softplus PWL Coefficients (64 Segments)...")

with open("softplus_pwl_coeffs.mem", "w") as f:
    # Segment width = 16.0 / 64 = 0.25
    step = 0.25
    
    for i in range(NUM_SEGMENTS):
        # i là index của Address (6 bit)
        # Logic map address sang giá trị thực (xem giải thích ở script SiLU của cậu)
        # Nếu i < 32 (Bit dấu = 0) -> x dương: 0.0, 0.25...
        # Nếu i >= 32 (Bit dấu = 1) -> x âm: -8.0, -7.75...
        
        if i < 32:
            start_val = i * step
        else:
            start_val = (i - 64) * step
            
        end_val = start_val + step
        
        # Lấy mẫu để Linear Regression
        x_points = np.linspace(start_val, end_val, 20)
        y_points = softplus(x_points)
        
        # Fit y = ax + b
        slope, intercept = np.polyfit(x_points, y_points, 1)
        
        # Convert Fixed Point
        a_fixed = float_to_fixed(slope)
        b_fixed = float_to_fixed(intercept)
        
        # Saturation
        a_fixed = max(min(a_fixed, 32767), -32768)
        b_fixed = max(min(b_fixed, 32767), -32768)
        
        # Ghi file: Slope (16bit) + Intercept (16bit)
        line = f"{to_hex(a_fixed)}{to_hex(b_fixed)}"
        f.write(line + "\n")
        
        # Debug
        if i == 0: print(f"Seg 0 [0.0 to 0.25]: y ≈ {slope:.3f}x + {intercept:.3f}")
        if i == 32: print(f"Seg 32 [-8.0 to -7.75]: y ≈ {slope:.3f}x + {intercept:.3f}")

print("Done! Saved to softplus_pwl_coeffs.mem")