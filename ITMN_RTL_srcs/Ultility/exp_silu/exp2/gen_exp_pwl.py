import numpy as np

# --- CẤU HÌNH ---
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
NUM_SEGMENTS = 64  
# Input range Q3.12: -8.0 to +7.999

def func_target(x):
    # Hàm mục tiêu: exp(x)
    # Lưu ý: Exp tăng rất nhanh. Với Q3.12 max = 7.99.
    # exp(7.99) ~ 2980. Vẫn nằm trong range signed 16-bit (max 32767 ~ 7.99)??? 
    # KHOAN! 
    # Q3.12 có range thực [-8.0, 7.99].
    # Giá trị thực 7.99 được biểu diễn là 32767.
    # Nhưng exp(7.99) = 2980 (giá trị thực).
    # Để biểu diễn số 2980 trong Q3.12 -> Cần 2980 * 4096 = 12,206,080 -> TRÀN 16-bit.
    
    # => Exp Unit Q3.12 CHỈ CÓ THỂ BIỂU DIỄN kết quả exp(x) khi exp(x) < 8.0.
    # Tức là x < ln(8.0) ~ 2.079.
    # Nếu input > 2.079, output sẽ bão hòa tại 7.99 (MAX_INT).
    # (Trừ khi output dùng định dạng khác input, nhưng module trên input/output cùng width).
    
    return np.exp(x)

def to_hex(val, width=16):
    val = int(val)
    if val < 0: val = (1 << width) + val
    return f"{val & ((1<<width)-1):04x}"

def float_to_fixed(val):
    return int(round(val * SCALE))

print("Generating EXP PWL Coefficients...")

step = 16.0 / NUM_SEGMENTS # 0.25

with open("exp_pwl_coeffs.mem", "w") as f:
    for i in range(NUM_SEGMENTS):
        # Map index 6-bit sang giá trị thực
        # 0..31 -> 0.0 .. 7.75
        # 32..63 -> -8.0 .. -0.25
        if i < 32:
            start_val = i * step
        else:
            start_val = (i - 64) * step
            
        end_val = start_val + step
        
        # Linear Regression
        x_points = np.linspace(start_val, end_val, 20)
        y_points = func_target(x_points)
        
        slope, intercept = np.polyfit(x_points, y_points, 1)
        
        # Convert to Fixed Point
        a_fixed = float_to_fixed(slope)
        b_fixed = float_to_fixed(intercept)
        
        # Saturation 16-bit
        a_fixed = max(min(a_fixed, 32767), -32768)
        b_fixed = max(min(b_fixed, 32767), -32768)
        
        # Ghi file
        line = f"{to_hex(a_fixed)}{to_hex(b_fixed)}"
        f.write(line + "\n")
        
        if i == 0: print(f"Seg 0 (0.0 to 0.25): y={slope:.2f}x+{intercept:.2f}")
        if i == 32: print(f"Seg 32 (-8.0 to -7.75): y={slope:.2f}x+{intercept:.2f}")

print("Done! Saved to exp_pwl_coeffs.mem")