import numpy as np

# --- CẤU HÌNH Q3.12 ---
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768
NUM_SEGMENTS = 64 # Số đoạn PWL

# --- ĐƯỜNG DẪN FILE ---
# File Hex đầu ra của Linear Proj 2 (Nhánh Z - Gate)
# Shape: (128000 dòng) hoặc tương tự
FILE_LINEAR_Z_HEX = "D:/DoAn1/Ultility/goldens/linear2_golden.mem" 

# File Output (Dùng làm input Gate cho Scan Core)
FILE_GATE_OUT     = "scan_gate_real.txt"

# --- HÀM HỖ TRỢ ---
def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def hex_to_int(hex_str):
    val = int(hex_str, 16)
    if val >= 32768: val -= 65536
    return val

def float_to_fixed(val):
    return int(round(val * SCALE))

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

# --- 1. TÁI TẠO BẢNG HỆ SỐ PWL (GIỐNG HỆT PHẦN CỨNG) ---
# Mục đích: Để Python tính ra kết quả y hệt module SiLU_Unit
print("1. Re-generating SiLU PWL Coefficients...")

def silu_func(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

pwl_rom = [] # Lưu (slope, intercept) cho 64 đoạn

step = 0.25 # 16.0 / 64
for i in range(NUM_SEGMENTS):
    # Logic xác định khoảng giá trị y hệt script gen_silu_pwl.py
    if i < 32: start_val = i * step
    else:      start_val = (i - 64) * step
    end_val = start_val + step
    
    # Linear Regression để tìm a, b
    x_points = np.linspace(start_val, end_val, 20)
    y_points = silu_func(x_points)
    slope, intercept = np.polyfit(x_points, y_points, 1)
    
    a_fixed = sat16(float_to_fixed(slope))
    b_fixed = sat16(float_to_fixed(intercept))
    
    pwl_rom.append((a_fixed, b_fixed))

# --- 2. HÀM MÔ PHỎNG HARDWARE SILU ---
def hardware_silu(in_val):
    # in_val là số nguyên Q3.12 (có dấu)
    
    # 1. Tính địa chỉ Segment (6 bit cao nhất)
    # Lấy raw bits, dịch phải 10, lấy 6 bit
    # Ví dụ: 0x1000 (1.0) -> 0001 00... -> >>10 = 4. Addr = 4.
    # Ví dụ: 0xF000 (-1.0) -> 1111 00... -> >>10 = 111100 (60). Addr = 60.
    # Trong Python phải xử lý bit cẩn thận
    raw_bits = in_val & 0xFFFF
    addr = (raw_bits >> 10) & 0x3F
    
    # 2. Tra bảng
    slope, intercept = pwl_rom[addr]
    
    # 3. Tính toán: (Slope * In) >> 12 + Intercept
    prod = (slope * in_val) >> FRAC_BITS
    res = prod + intercept
    
    # 4. Saturation
    return sat16(res)

# --- 3. XỬ LÝ FILE DỮ LIỆU ---
print(f"2. Processing {FILE_LINEAR_Z_HEX}...")

try:
    with open(FILE_LINEAR_Z_HEX, 'r') as f:
        z_hex_lines = [line.strip() for line in f if line.strip()]
        
    gate_out_lines = []
    
    for hex_val in z_hex_lines:
        # Convert Hex -> Int
        in_val = hex_to_int(hex_val)
        
        # Qua mô hình Hardware SiLU
        out_val = hardware_silu(in_val)
        
        # Convert Int -> Hex
        gate_out_lines.append(to_hex(out_val))
        
    # Ghi file
    with open(FILE_GATE_OUT, 'w') as f:
        f.write("\n".join(gate_out_lines))
        
    print(f"DONE! Processed {len(gate_out_lines)} values.")
    print(f"Saved to: {FILE_GATE_OUT}")
    print(">> Nạp file này vào 'mem_gate' trong Testbench Scan Core.")

except FileNotFoundError:
    print("ERROR: Không tìm thấy file input hex.")