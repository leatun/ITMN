import numpy as np
import math

# --- CONFIG ---
SEQ_LEN    = 1000
D_INNER    = 128      # 128 Kênh
D_STATE    = 16       # N = 16
DATA_WIDTH = 16
FRAC_BITS  = 12
SCALE      = 1 << FRAC_BITS

MAX_INT = (1 << (DATA_WIDTH - 1)) - 1
MIN_INT = -(1 << (DATA_WIDTH - 1))

# --- FIXED-POINT HELPERS ---
def sat16(x: int) -> int:
    if x > MAX_INT: return MAX_INT
    if x < MIN_INT: return MIN_INT
    return int(x)

def float_to_fixed(val):
    v = np.round(np.array(val, dtype=np.float64) * SCALE).astype(np.int64)
    v = np.clip(v, MIN_INT, MAX_INT).astype(np.int64)
    return v

def fixed_mul(a: int, b: int) -> int:
    prod = int(a) * int(b)
    prod >>= FRAC_BITS
    return sat16(prod)

def fixed_add(a: int, b: int) -> int:
    return sat16(int(a) + int(b))

def to_hex(val, width=16):
    v = int(val) & ((1 << width) - 1)
    return f"{v:0{width//4}x}"

def sigmoid_stable(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + math.exp(-x))
    else:      return math.exp(x) / (1.0 + math.exp(x))

def silu_fixed(x_fixed: int) -> int:
    x_real = int(x_fixed) / SCALE
    y_real = x_real * sigmoid_stable(x_real)
    return int(float_to_fixed(y_real))

# --- 1) GENERATE FULL MODEL DATA ---
print(f"Generating FULL Mamba Data (D={D_INNER})...")
np.random.seed(42)

# --- Channel-Specific Parameters (Full 128 channels) ---
# A: Shape (128, 16)
A_fixed = float_to_fixed(np.random.uniform(-1.0, -0.1, (D_INNER, D_STATE)))
# D: Shape (128,)
D_fixed = float_to_fixed(np.random.uniform(-1.0, 1.0, (D_INNER,)))

# Inputs: Shape (128, 1000)
delta_fixed = float_to_fixed(np.random.uniform(0.1, 1.0, (D_INNER, SEQ_LEN)))
x_fixed     = float_to_fixed(np.random.uniform(-1.0, 1.0, (D_INNER, SEQ_LEN)))
gate_fixed  = float_to_fixed(np.random.uniform(-2.0, 2.0, (D_INNER, SEQ_LEN)))

# --- Shared Parameters ---
# B_raw, C_raw: Shape (1000, 16) -
B_fixed = float_to_fixed(np.random.uniform(-0.5, 0.5, (SEQ_LEN, D_STATE)))
C_fixed = float_to_fixed(np.random.uniform(-0.5, 0.5, (SEQ_LEN, D_STATE)))

# --- 2) GOLDEN COMPUTE (ALL 128 CHANNELS) ---
Y_golden = np.zeros((D_INNER, SEQ_LEN), dtype=np.int64)

print("Computing Golden Output for all 128 channels...")

for d in range(D_INNER):
    h_state = np.zeros(D_STATE, dtype=np.int64)
    
    # Lấy tham số tĩnh của kênh d
    A_vec = A_fixed[d]
    D_val = int(D_fixed[d])

    for t in range(SEQ_LEN):
        delta_val = int(delta_fixed[d, t])
        x_val     = int(x_fixed[d, t])
        gate_val  = int(gate_fixed[d, t])
        
        # B và C dùng chung cho mọi kênh (lấy theo t)
        B_vec = B_fixed[t]
        C_vec = C_fixed[t]

        # --- SSM CORE LOGIC ---
        deltaA = np.zeros(D_STATE, dtype=np.int64)
        discA  = np.zeros(D_STATE, dtype=np.int64)
        deltaB = np.zeros(D_STATE, dtype=np.int64)
        deltaBx = np.zeros(D_STATE, dtype=np.int64)

        for n in range(D_STATE):
            deltaA[n] = fixed_mul(delta_val, int(A_vec[n]))
            real_val = deltaA[n] / SCALE
            exp_val = math.exp(real_val)
            discA[n] = int(float_to_fixed(exp_val))

            deltaB[n] = fixed_mul(delta_val, int(B_vec[n]))
            deltaBx[n] = fixed_mul(deltaB[n], x_val)

        h_new = np.zeros(D_STATE, dtype=np.int64)
        for n in range(D_STATE):
            term1 = fixed_mul(discA[n], int(h_state[n]))
            h_new[n] = fixed_add(term1, deltaBx[n])
        h_state = h_new

        y_scan = 0
        for n in range(D_STATE):
            prod = fixed_mul(int(C_vec[n]), int(h_state[n]))
            y_scan += prod 

        Dx = fixed_mul(D_val, x_val)
        y_with_D = y_scan + Dx

        g_act = silu_fixed(gate_val)
        y_final = (y_with_D * g_act) >> FRAC_BITS
        y_final = sat16(y_final)
        
        Y_golden[d, t] = y_final

# --- 3) WRITE FILES (FLATTENED) ---
def write_vec(fname, vec_data):
    with open(fname, "w") as f:
        if len(vec_data.shape) > 1:
            for i in range(vec_data.shape[0]):
                for j in range(vec_data.shape[1]):
                    f.write(to_hex(vec_data[i, j]) + "\n")
        else:
            for val in vec_data:
                f.write(to_hex(val) + "\n")

print("Writing FULL files...")

# A: (128, 16) -> 2048 dòng
write_vec("scan_A.txt", A_fixed) 
# D: (128,) -> 128 dòng
write_vec("scan_D.txt", D_fixed) 

# Inputs: (128, 1000) -> 128000 dòng
# Lưu ý: Channel 0 nằm đầu tiên (dòng 0-999)
write_vec("scan_delta.txt", delta_fixed)
write_vec("scan_x.txt",     x_fixed)
write_vec("scan_gate.txt",  gate_fixed)

# B, C: (1000, 16) -> 16000 dòng (Shared)
write_vec("scan_B_full.txt", B_fixed)
write_vec("scan_C_full.txt", C_fixed)

# Golden Output: (128, 1000) -> 128000 dòng
write_vec("scan_y_golden.txt", Y_golden)

print("Done! Files generated with FULL SHAPES.")
print(f"A: {A_fixed.shape}, D: {D_fixed.shape}")
print(f"B: {B_fixed.shape}, C: {C_fixed.shape}")
print("Testbench will automatically simulate Channel 0 (first lines of files).")