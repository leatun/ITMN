import numpy as np
import math

# --- CONFIG ---
SEQ_LEN   = 1000
D_INNER   = 128      # <<<<< set theo project (vd 64 / 128)
D_STATE   = 16
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

MAX_INT = (1 << (DATA_WIDTH - 1)) - 1
MIN_INT = -(1 << (DATA_WIDTH - 1))

# --- FIXED-POINT HELPERS (Q3.12) ---
def sat16(x: int) -> int:
    if x > MAX_INT: return MAX_INT
    if x < MIN_INT: return MIN_INT
    return int(x)

def float_to_fixed(val):
    # val: float or np array
    v = np.round(np.array(val, dtype=np.float64) * SCALE).astype(np.int64)
    v = np.clip(v, MIN_INT, MAX_INT).astype(np.int64)
    return v

def fixed_mul(a: int, b: int) -> int:
    # (a*b)>>FRAC_BITS with sat16
    prod = int(a) * int(b)
    prod >>= FRAC_BITS
    return sat16(prod)

def fixed_add(a: int, b: int) -> int:
    return sat16(int(a) + int(b))

def to_hex(val, width=16):
    # write as 16-bit two's complement hex
    v = int(val) & ((1 << width) - 1)
    return f"{v:0{width//4}x}"

def sigmoid_stable(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def silu_fixed(x_fixed: int) -> int:
    # SiLU(x) = x * sigmoid(x)  (compute in float then quantize)
    x_real = int(x_fixed) / SCALE
    y_real = x_real * sigmoid_stable(x_real)
    return int(float_to_fixed(y_real))

# --- 1) GENERATE RANDOM WEIGHTS / INPUTS ---
print("Generating Full Scan Core Data (SSM + D + Gate) with D_INNER channels...")
np.random.seed(999)

# A,B,C: per-channel per-state (D_INNER, D_STATE)
A_fixed = float_to_fixed(np.random.uniform(-1.0, -0.1, (D_INNER, D_STATE)))
B_fixed = float_to_fixed(np.random.uniform(-0.5,  0.5, (D_INNER, D_STATE)))
C_fixed = float_to_fixed(np.random.uniform(-0.5,  0.5, (D_INNER, D_STATE)))

# D: per-channel vector (D_INNER,)
D_fixed = float_to_fixed(np.random.uniform(-1.0, 1.0, (D_INNER,)))

# Inputs: per-channel per-time
delta_fixed = float_to_fixed(np.random.uniform(0.1, 1.0, (D_INNER, SEQ_LEN)))
x_fixed     = float_to_fixed(np.random.uniform(-1.0, 1.0, (D_INNER, SEQ_LEN)))
gate_fixed  = float_to_fixed(np.random.uniform(-2.0, 2.0, (D_INNER, SEQ_LEN)))

# --- 2) GOLDEN COMPUTE ---
# h_state: per-channel state vector
h_state = np.zeros((D_INNER, D_STATE), dtype=np.int64)

# y_golden: (D_INNER, SEQ_LEN)
Y_golden = np.zeros((D_INNER, SEQ_LEN), dtype=np.int64)

for d in range(D_INNER):
    for t in range(SEQ_LEN):

        # --- SSM discretization (same style as your old script) ---
        # deltaA[n] = delta * A[n]
        deltaA = np.zeros(D_STATE, dtype=np.int64)
        discA  = np.zeros(D_STATE, dtype=np.int64)
        deltaB = np.zeros(D_STATE, dtype=np.int64)
        deltaBx = np.zeros(D_STATE, dtype=np.int64)

        for n in range(D_STATE):
            deltaA[n] = fixed_mul(int(delta_fixed[d, t]), int(A_fixed[d, n]))

            # discA = exp(deltaA)
            real_val = deltaA[n] / SCALE
            exp_val = math.exp(real_val)  # safe because A is negative range
            discA[n] = int(float_to_fixed(exp_val))

            deltaB[n] = fixed_mul(int(delta_fixed[d, t]), int(B_fixed[d, n]))
            deltaBx[n] = fixed_mul(int(deltaB[n]), int(x_fixed[d, t]))

        # h_new[n] = discA[n]*h + deltaBx[n]
        h_new = np.zeros(D_STATE, dtype=np.int64)
        for n in range(D_STATE):
            term1 = fixed_mul(int(discA[n]), int(h_state[d, n]))
            h_new[n] = fixed_add(int(term1), int(deltaBx[n]))
        h_state[d, :] = h_new

        # y_scan = sum_n C[n]*h[n]
        # keep wide accumulator (python int), clamp later
        y_scan = 0
        for n in range(D_STATE):
            prod = fixed_mul(int(C_fixed[d, n]), int(h_state[d, n]))
            y_scan += int(prod)  # not saturated each add (same spirit as old script)

        # --- Add residual D*x ---
        Dx = fixed_mul(int(D_fixed[d]), int(x_fixed[d, t]))
        y_with_D = y_scan + int(Dx)

        # --- Gate: SiLU(gate[d,t]) ---
        g_act = silu_fixed(int(gate_fixed[d, t]))

        # y_final = (y_with_D * g_act) >> FRAC_BITS
        y_final = (int(y_with_D) * int(g_act)) >> FRAC_BITS

        # final saturation
        y_final = sat16(y_final)
        Y_golden[d, t] = y_final

# --- 3) WRITE FILES ---
# Flatten order used here:
# - matrices saved as: d=0: all t (or all state), then d=1, ...
# This matches C layout [D_INNER][SEQ_LEN] (row-major by inner dimension index first)

def write_mat_DINNERxN(fname, mat_2d):
    with open(fname, "w") as f:
        for d in range(mat_2d.shape[0]):
            for j in range(mat_2d.shape[1]):
                f.write(to_hex(mat_2d[d, j]) + "\n")

def write_vec(fname, vec_1d):
    with open(fname, "w") as f:
        for d in range(vec_1d.shape[0]):
            f.write(to_hex(vec_1d[d]) + "\n")

write_mat_DINNERxN("scan_A.txt", A_fixed)
write_mat_DINNERxN("scan_B.txt", B_fixed)
write_mat_DINNERxN("scan_C.txt", C_fixed)

write_mat_DINNERxN("scan_delta.txt", delta_fixed)
write_mat_DINNERxN("scan_x.txt", x_fixed)

write_vec("scan_D.txt", D_fixed)
write_mat_DINNERxN("scan_gate.txt", gate_fixed)

write_mat_DINNERxN("scan_y_golden.txt", Y_golden)

print("Done! All files generated with shapes:")
print(f"  A,B,C    : ({D_INNER}, {D_STATE})")
print(f"  delta,x,gate,y : ({D_INNER}, {SEQ_LEN})")
print(f"  D        : ({D_INNER},)")
