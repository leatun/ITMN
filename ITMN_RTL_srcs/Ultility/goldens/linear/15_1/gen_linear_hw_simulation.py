import numpy as np
import random

# --- CẤU HÌNH ---
D_MODEL = 64     # Input Dim
D_INNER = 128    # Output Dim
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
# Giới hạn 16-bit Signed
MAX_INT = 32767
MIN_INT = -32768

# File Output
FILE_X = "lin_hw_x.txt"
FILE_W = "lin_hw_w.txt"
FILE_Y = "lin_hw_y.txt"

# --- HÀM MÔ PHỎNG PHẦN CỨNG (QUAN TRỌNG) ---
def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def hw_mac(acc, x, w):
    # 1. Nhân
    prod = int(x) * int(w)
    # 2. Dịch bit (Arithmetic Shift Right)
    prod_shifted = prod >> FRAC_BITS
    # 3. Cộng dồn
    res = int(acc) + int(prod_shifted)
    # 4. Bão hòa (Unified PE bão hòa sau mỗi lần cộng)
    return sat16(res)

def run():
    print("=== GENERATING HARDWARE SIMULATION DATA ===")
    
    # 1. TẠO DỮ LIỆU NGẪU NHIÊN (Q3.12 Integers)
    # X: 64 giá trị
    # Range: -2.0 đến 2.0 (khoảng -8192 đến 8192)
    X_int = [random.randint(-8000, 8000) for _ in range(D_MODEL)]
    
    # W: 128 hàng x 64 cột
    # Range nhỏ để tránh bão hòa sớm: -0.5 đến 0.5 (-2000 đến 2000)
    W_int = [[random.randint(-2000, 2000) for _ in range(D_MODEL)] for _ in range(D_INNER)]
    
    # 2. TÍNH TOÁN GOLDEN OUTPUT (THEO LOGIC PE)
    print("Simulating Hardware Logic...")
    Y_int = []
    
    # Duyệt từng kênh Output (0..127)
    for o in range(D_INNER):
        acc = 0 # Bias = 0
        
        # Duyệt từng phần tử Input (0..63)
        for i in range(D_MODEL):
            x_val = X_int[i]
            w_val = W_int[o][i] # Hàng o, Cột i
            
            # Tính MAC y hệt Unified_PE
            acc = hw_mac(acc, x_val, w_val)
            
        Y_int.append(acc)

    # 3. XUẤT FILE X
    print(f"Exporting {FILE_X}...")
    with open(FILE_X, "w") as f:
        for val in X_int:
            f.write(to_hex(val) + "\n")

    # 4. XUẤT FILE W (REORDERING CHO CONTROLLER)
    # Controller đọc: Chunk 0 (0..15), Feed 0 (Cột 0) -> Feed 63 (Cột 63)
    print(f"Exporting {FILE_W} (Reordered)...")
    hw_w_list = []
    
    num_chunks = D_INNER // 16 # 8
    for chunk in range(num_chunks):
        start_row = chunk * 16
        end_row = start_row + 16
        
        # Duyệt theo Cột (Input Feed) trước
        for col in range(D_MODEL):
            # Lấy 16 weight của cột này cho 16 hàng trong chunk
            for row in range(start_row, end_row):
                val = W_int[row][col]
                hw_w_list.append(val)
                
    with open(FILE_W, "w") as f:
        for val in hw_w_list:
            f.write(to_hex(val) + "\n")

    # 5. XUẤT FILE Y
    print(f"Exporting {FILE_Y}...")
    with open(FILE_Y, "w") as f:
        for val in Y_int:
            f.write(to_hex(val) + "\n")
            
    print("DONE! Data generated successfully.")

if __name__ == "__main__":
    run()