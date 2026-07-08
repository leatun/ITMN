import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_MODEL = 64
OUT_CHANNELS = 16 # Số lượng PE của cậu
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

MAX_INT = (1 << (DATA_WIDTH - 1)) - 1
MIN_INT = -(1 << (DATA_WIDTH - 1))

# --- HÀM HỖ TRỢ ---
def float_to_fixed(val):
    # Nhân scale và làm tròn
    val = np.round(val * SCALE).astype(int)
    # Clamp (Bão hòa đầu vào nếu cần, dù thường mình control được input)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def to_hex(val, width=16):
    # Xử lý bù 2 để in ra hex đúng
    val = int(val)
    if val < 0:
        val = (1 << width) + val
    return f"{val:04x}" # 4 ký tự hex cho 16 bit

# --- 1. TẠO DỮ LIỆU ---
print(f"Generating data for SEQ={SEQ_LEN}, D_MODEL={D_MODEL}, PEs={OUT_CHANNELS}...")
np.random.seed(42)

# Input X (SEQ_LEN, D_MODEL) - Random từ -1.0 đến 1.0
X_float = np.random.uniform(-1.0, 1.0, (SEQ_LEN, D_MODEL))
X_fixed = float_to_fixed(X_float)

# Weights W (OUT_CHANNELS, D_MODEL) - Random từ -0.5 đến 0.5
W_float = np.random.uniform(-0.5, 0.5, (OUT_CHANNELS, D_MODEL))
W_fixed = float_to_fixed(W_float)

# Bias b (OUT_CHANNELS) - Random từ -0.1 đến 0.1
b_float = np.random.uniform(-0.1, 0.1, (OUT_CHANNELS,))
b_fixed = float_to_fixed(b_float)

# --- 2. TÍNH TOÁN GOLDEN OUTPUT (MÔ PHỎNG PHẦN CỨNG) ---
# Chúng ta không dùng matmul của numpy trực tiếp vì cần mô phỏng 
# việc dịch bit và bão hòa sau mỗi bước cộng dồn nếu cần (nhưng ở đây PE dịch sau nhân)

Y_fixed = np.zeros((SEQ_LEN, OUT_CHANNELS), dtype=int)

print("Calculating Golden Output...")
for t in range(SEQ_LEN):
    for o in range(OUT_CHANNELS):
        acc = 0
        for i in range(D_MODEL):
            # Mô phỏng: Nhân -> 32bit -> Dịch phải
            mult_res = int(X_fixed[t, i]) * int(W_fixed[o, i])
            mult_shifted = mult_res >> FRAC_BITS # Dịch bit số học
            acc += mult_shifted
        
        # Cộng Bias
        acc += b_fixed[o]
        
        # Mô phỏng Saturation (Chống tràn đầu ra)
        if acc > MAX_INT: acc = MAX_INT
        if acc < MIN_INT: acc = MIN_INT
        
        Y_fixed[t, o] = acc

# --- 3. XUẤT FILE TXT ---
print("Saving to .txt files...")

# File x_input.txt: Mỗi dòng 1 giá trị hex (tổng 1000 * 64 dòng)
with open("x_input.txt", "w") as f:
    for t in range(SEQ_LEN):
        for i in range(D_MODEL):
            f.write(to_hex(X_fixed[t, i]) + "\n")

# File W_input.txt: Sắp xếp theo cột để tiện nạp vào 16 PE
# Cấu trúc: Cột 0 (16 dòng), Cột 1 (16 dòng)...
# Tổng: 64 cột * 16 hàng = 1024 dòng
with open("W_input.txt", "w") as f:
    for i in range(D_MODEL):     # Duyệt theo cột trước
        for o in range(OUT_CHANNELS): # Duyệt qua các hàng (PE)
            f.write(to_hex(W_fixed[o, i]) + "\n")

# File b_input.txt: 16 giá trị
with open("b_input.txt", "w") as f:
    for o in range(OUT_CHANNELS):
        f.write(to_hex(b_fixed[o]) + "\n")

# File y_golden.txt: Để so sánh (tổng 1000 * 16 dòng)
with open("y_golden.txt", "w") as f:
    for t in range(SEQ_LEN):
        for o in range(OUT_CHANNELS):
            f.write(to_hex(Y_fixed[t, o]) + "\n")

print("Done! Files created: x_input.txt, W_input.txt, b_input.txt, y_golden.txt")