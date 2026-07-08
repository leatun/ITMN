import numpy as np
import math

# --- CẤU HÌNH ---
SEQ_LEN = 1000 # Test step
D_STATE = 16
DATA_WIDTH = 16
FRAC_BITS = 12 # Q3.12
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
    if val < 0: val = (1 << width) + val
    return f"{val:04x}"

def fixed_mul(a, b):
    # Mô phỏng Unified_PE: Nhân -> Dịch bit -> Bão hòa
    res = (a * b) >> FRAC_BITS
    if res > MAX_INT: res = MAX_INT
    if res < MIN_INT: res = MIN_INT
    return res

def fixed_add(a, b):
    res = a + b
    if res > MAX_INT: res = MAX_INT
    if res < MIN_INT: res = MIN_INT
    return res

# --- 1. TẠO DỮ LIỆU ---
print("Generating Scan Core Data...")
np.random.seed(42)

# Parameter A (D_STATE) - Luôn âm để exp ổn định
A_float = np.random.uniform(-1.0, -0.1, (D_STATE,))
A_fixed = float_to_fixed(A_float)

# Parameter B (D_STATE)
B_float = np.random.uniform(-0.5, 0.5, (D_STATE,))
B_fixed = float_to_fixed(B_float)

# Parameter C (D_STATE)
C_float = np.random.uniform(-0.5, 0.5, (D_STATE,))
C_fixed = float_to_fixed(C_float)

# Input Delta (SEQ_LEN) - Luôn dương
delta_float = np.random.uniform(0.1, 1.0, (SEQ_LEN,))
delta_fixed = float_to_fixed(delta_float)

# Input X (SEQ_LEN)
x_float = np.random.uniform(-1.0, 1.0, (SEQ_LEN,))
x_fixed = float_to_fixed(x_float)

# --- 2. TÍNH TOÁN GOLDEN OUTPUT (Hardware Logic) ---
# h khởi tạo bằng 0
h_state = np.zeros(D_STATE, dtype=int)
Y_golden = []

print("\n--- Simulation Trace ---")
for t in range(SEQ_LEN):
    # 1. Tính DeltaA = Delta * A
    deltaA = np.array([fixed_mul(delta_fixed[t], A_fixed[n]) for n in range(D_STATE)])
    
    # 2. Tính DiscA = Exp(DeltaA)
    # Lưu ý: Mô phỏng lại bảng tra ROM của cậu
    # (Ở đây dùng math.exp cho gần đúng, phần cứng có thể lệch 1-2 đơn vị do làm tròn ROM)
    discA = []
    for val in deltaA:
        real_val = val / SCALE
        exp_val = math.exp(real_val)
        discA.append(float_to_fixed(exp_val))
    discA = np.array(discA)

    # 3. Tính DeltaB = Delta * B
    deltaB = np.array([fixed_mul(delta_fixed[t], B_fixed[n]) for n in range(D_STATE)])

    # 4. Tính DeltaBx = DeltaB * x
    deltaBx = np.array([fixed_mul(deltaB[n], x_fixed[t]) for n in range(D_STATE)])

    # 5. Cập nhật h: h_new = (discA * h_old) + deltaBx
    h_new = np.zeros(D_STATE, dtype=int)
    for n in range(D_STATE):
        term1 = fixed_mul(discA[n], h_state[n])
        h_new[n] = fixed_add(term1, deltaBx[n])
    
    # Debug: In h[0] để so sánh với Waveform
    if t < 5:
        print(f"Time {t}: h[0] updated to {h_new[0]} (Hex: {to_hex(h_new[0])})")

    # Cập nhật trạng thái
    h_state = h_new

    # 6. Tính Output y = Sum(C * h)
    y_val = 0
    for n in range(D_STATE):
        prod = fixed_mul(C_fixed[n], h_state[n])
        y_val += prod # Adder tree cộng dồn (thường không dịch bit, nhưng cẩn thận tràn)
    
    # Bão hòa output cuối cùng
    if y_val > MAX_INT: y_val = MAX_INT
    if y_val < MIN_INT: y_val = MIN_INT
    
    Y_golden.append(y_val)

# --- 3. XUẤT FILE ---
# Ghi A, B, C (1 dòng mỗi loại, các phần tử cách nhau bởi space hoặc newline)
# Để dễ đọc bằng $readmemh, ta ghi mỗi giá trị 1 dòng
with open("scan_A.txt", "w") as f:
    for v in A_fixed: f.write(to_hex(v) + "\n")
with open("scan_B.txt", "w") as f:
    for v in B_fixed: f.write(to_hex(v) + "\n")
with open("scan_C.txt", "w") as f:
    for v in C_fixed: f.write(to_hex(v) + "\n")

# Ghi Delta và X (theo thời gian)
with open("scan_delta.txt", "w") as f:
    for v in delta_fixed: f.write(to_hex(v) + "\n")
with open("scan_x.txt", "w") as f:
    for v in x_fixed: f.write(to_hex(v) + "\n")

# Ghi Y Golden
with open("scan_y_golden.txt", "w") as f:
    for v in Y_golden: f.write(to_hex(v) + "\n")

print("Done! Files generated.")