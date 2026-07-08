import numpy as np

# --- CẤU HÌNH CHO FULL MODEL (in_proj1) ---
SEQ_LEN = 1000
D_MODEL = 64            # Input Dimension (IN_DIM)
EXPAND = 2
D_INNER = D_MODEL * EXPAND  # Output Dimension (OUT_DIM) = 128

# Cấu hình phần cứng (để biết logic fixed-point)
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

MAX_INT = (1 << (DATA_WIDTH - 1)) - 1
MIN_INT = -(1 << (DATA_WIDTH - 1))

# --- HÀM HỖ TRỢ ---
def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def to_hex(val, width=16):
    val = int(val)
    if val < 0:
        val = (1 << width) + val
    return f"{val:04x}"

# --- 1. TẠO DỮ LIỆU ---
print(f"Generating Linear data: SEQ={SEQ_LEN}, IN={D_MODEL}, OUT={D_INNER}...")
np.random.seed(123) # Seed mới cho may mắn

# Input X (SEQ_LEN, D_MODEL)
X_float = np.random.uniform(-1.0, 1.0, (SEQ_LEN, D_MODEL))
X_fixed = float_to_fixed(X_float)

# Weights W (D_INNER, D_MODEL) - Shape (128, 64)
# Hàng là Output Channel, Cột là Input Channel
W_float = np.random.uniform(-0.2, 0.2, (D_INNER, D_MODEL)) # Giảm range để tránh tràn số khi cộng nhiều
W_fixed = float_to_fixed(W_float)

# Bias b (D_INNER) - Shape (128,)
b_float = np.random.uniform(-0.1, 0.1, (D_INNER,))
b_fixed = float_to_fixed(b_float)

# --- 2. TÍNH TOÁN GOLDEN OUTPUT (MÔ PHỎNG PHẦN CỨNG) ---
# Y = X * W^T + b
Y_fixed = np.zeros((SEQ_LEN, D_INNER), dtype=int)

print("Calculating Golden Output (Hardware Logic)...")
for t in range(SEQ_LEN):
    for o in range(D_INNER): # 0 -> 127
        acc = 0
        for i in range(D_MODEL): # 0 -> 63
            # Mô phỏng Unified PE: Nhân -> Dịch bit ngay
            mult_res = int(X_fixed[t, i]) * int(W_fixed[o, i])
            mult_shifted = mult_res >> FRAC_BITS 
            acc += mult_shifted
        
        # Cộng Bias
        acc += int(b_fixed[o])
        
        # Saturation
        if acc > MAX_INT: acc = MAX_INT
        if acc < MIN_INT: acc = MIN_INT
        
        Y_fixed[t, o] = acc

# --- 3. XUẤT FILE TXT ---
print("Saving to .txt files...")

# 1. File x_linear_input.txt (1000 dòng, mỗi dòng 1 giá trị hex? Hay 1 dòng 1 vector?)
# Để đơn giản cho testbench đọc: Ghi từng giá trị một xuống dòng
# Thứ tự: T0_I0, T0_I1... T0_I63, T1_I0...
with open("x_linear_input.txt", "w") as f:
    for t in range(SEQ_LEN):
        for i in range(D_MODEL):
            f.write(to_hex(X_fixed[t, i]) + "\n")

# 2. File w_linear_input.txt
# Cấu trúc này quan trọng cho Testbench.
# Nếu Linear Accelerator của cậu nạp theo cột (để 16 PE cùng nhân với 1 giá trị x):
# Cần ghi: Cột 0 (Hàng 0..127), Cột 1 (Hàng 0..127)...
# NHƯNG: Hardware cậu chỉ có 16 PE. Nên cậu sẽ xử lý 16 hàng một lúc.
# Để tổng quát, ta cứ ghi theo thứ tự: [Cột 0 của tất cả hàng], [Cột 1 của tất cả hàng]...
with open("w_linear_input.txt", "w") as f:
    for i in range(D_MODEL):      # Duyệt theo cột input (64 cột)
        for o in range(D_INNER):  # Duyệt theo hàng output (128 hàng)
            f.write(to_hex(W_fixed[o, i]) + "\n")

# 3. File b_linear_input.txt (128 dòng)
with open("b_linear_input.txt", "w") as f:
    for o in range(D_INNER):
        f.write(to_hex(b_fixed[o]) + "\n")

# 4. File y_linear_golden.txt (1000 * 128 dòng)
# Thứ tự: T0_O0...T0_O127, T1_O0...
with open("y_linear_golden.txt", "w") as f:
    for t in range(SEQ_LEN):
        for o in range(D_INNER):
            f.write(to_hex(Y_fixed[t, o]) + "\n")

print(f"Done! Files created for D_MODEL={D_MODEL} -> D_INNER={D_INNER}.")
print(f"Weight file lines: {D_MODEL * D_INNER}")