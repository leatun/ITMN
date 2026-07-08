import numpy as np
import math
import re

# ==============================================================================
# 1. CẤU HÌNH & HELPER
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE 
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/" # Ví dụ
DIR_VEC    = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"

# Input Raw
FILE_A_LOG = DIR_VEC + "A_log.txt"
FILE_D     = DIR_VEC + "D.txt"

# Input Hex (Đã generate từ các bước trước)
FILE_X     = "conv_y_golden_ptb.txt"      # Channel-First
FILE_GATE  = "linear2_golden.txt"         # Token-First (Verified)
FILE_B     = "scan_real_B_shared.txt"     # Token-First (Verified)
FILE_C     = "scan_real_C_shared.txt"     # Token-First (Verified)
FILE_DELTA = "D:/DoAn1/Ultility/goldens/linear/15_1/xproj/gold_delta_final.txt"       # Token-First (Verified via script)

# Output
OUT_A_HEX  = "scan_A.txt"
OUT_D_HEX  = "scan_D.txt"
OUT_Y_GOLD = "gold_scan_final.txt"

def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def to_hex(val):
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def float_to_fixed(val):
    return int(round(val * SCALE))

def fixed_mul(a, b):
    # Mô phỏng nhân Fixed-point Q3.12
    return (a * b) >> FRAC_BITS

def silu_fixed_approx(x_fixed):
    # SiLU = x * sigmoid(x)
    # Dùng float để mô phỏng (sai số nhỏ chấp nhận được so với PWL)
    x_real = x_fixed / SCALE
    sigmoid = 1.0 / (1.0 + math.exp(-x_real))
    return int(x_fixed * sigmoid)

# ==============================================================================
# 2. LOAD & PROCESS DATA
# ==============================================================================
def load_data():
    print("--- Loading Data ---")

    # 1. Process A & D (Raw Float -> Fixed)
    print(f"Processing A_log and D...")
    A_fixed = np.zeros((D_INNER, D_STATE), dtype=int)
    D_fixed = np.zeros(D_INNER, dtype=int)
    
    # Load A_log
    try:
        with open(FILE_A_LOG) as f:
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    val_real = -math.exp(v)
                    
                    # --- THÊM DÒNG NÀY: MÔ PHỎNG HARDWARE CLAMPING ---
                    # Ép A phải nằm trong vùng Q3.12 [-8.0, 7.99]
                    # Để Golden File khớp với thực tế phần cứng
                    val_real = max(-8.0, min(7.999, val_real)) 
                    
                    A_fixed[r, c] = float_to_fixed(val_real)
    except Exception as e: print(f"Err A: {e}")

    # Load D
    try:
        with open(FILE_D) as f:
            idx = 0
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 2: D_fixed[int(parts[0])] = float_to_fixed(float(parts[1]))
                elif len(parts) == 1: 
                    D_fixed[idx] = float_to_fixed(float(parts[0]))
                    idx += 1
    except Exception as e: print(f"Err D: {e}")

    # 2. Load Hex Files
    print(f"Loading Hex Files...")
    
    # X: Channel-First -> (128, 1000)
    with open(FILE_X) as f: x_dat = [to_signed(l.strip()) for l in f if l.strip()]
    X = np.array(x_dat).reshape(D_INNER, SEQ_LEN)
    
    # Gate: Token-First -> (1000, 128)
    with open(FILE_GATE) as f: g_dat = [to_signed(l.strip()) for l in f if l.strip()]
    Gate = np.array(g_dat).reshape(SEQ_LEN, D_INNER)
    
    # Delta: Token-First -> (1000, 128)
    with open(FILE_DELTA) as f: d_dat = [to_signed(l.strip()) for l in f if l.strip()]
    Delta = np.array(d_dat).reshape(SEQ_LEN, D_INNER)
    
    # B, C: Token-First -> (1000, 16)
    with open(FILE_B) as f: b_dat = [to_signed(l.strip()) for l in f if l.strip()]
    B = np.array(b_dat).reshape(SEQ_LEN, D_STATE)
    
    with open(FILE_C) as f: c_dat = [to_signed(l.strip()) for l in f if l.strip()]
    C = np.array(c_dat).reshape(SEQ_LEN, D_STATE)

    return A_fixed, D_fixed, X, Gate, Delta, B, C

# ==============================================================================
# 3. RUN SCAN SIMULATION
# ==============================================================================
def run():
    A, D, X, Gate, Delta, B, C = load_data()
    
    # Save A & D for Testbench
    with open(OUT_A_HEX, 'w') as f:
        for val in A.flatten(): f.write(to_hex(val) + "\n")
    with open(OUT_D_HEX, 'w') as f:
        for val in D: f.write(to_hex(val) + "\n")
        
    print("--- Running Scan Core Simulation (Hardware Logic) ---")
    Y_final_list = []
    
    # Hardware Scan Core chạy theo Channel-First (Chunk Loop -> Time Loop)
    for d in range(D_INNER):
        # Reset State cho mỗi kênh
        h = np.zeros(D_STATE, dtype=int)
        
        # Static Params
        A_row = A[d]
        D_val = D[d]
        
        for t in range(SEQ_LEN):
            # Lấy Input
            u_val = X[d, t]          # X: [Channel, Time]
            delta = Delta[t, d]      # Delta: [Time, Channel] (Do file token-first)
            gate  = Gate[t, d]       # Gate:  [Time, Channel] (Do file token-first)
            
            B_t = B[t]               # B: [Time, State]
            C_t = C[t]               # C: [Time, State]
            
            # --- SSM Calculation (Bit-Exact Logic) ---
            
            # 1. Discretization A: dA = exp(delta * A)
            # Hardware dùng Exp Unit
            disc_A = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                prod = fixed_mul(delta, A_row[n])
                # Exp float approx
                val_real = prod / SCALE
                disc_A[n] = float_to_fixed(math.exp(val_real))
            
            # 2. Discretization B: dBx = (delta * B) * u
            dBx = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                dB = fixed_mul(delta, B_t[n])
                dBx[n] = fixed_mul(dB, u_val)
                
            # 3. Update State: h = disc_A * h + dBx
            h_new = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                term1 = fixed_mul(disc_A[n], h[n])
                res = term1 + dBx[n]
                # Saturation
                if res > MAX_INT: res = MAX_INT
                elif res < MIN_INT: res = MIN_INT
                h_new[n] = res
            h = h_new
            
            # 4. Output Projection: y = C * h
            y_scan = 0
            for n in range(D_STATE):
                y_scan += fixed_mul(C_t[n], h[n])
                
            # 5. Skip D: y = y + D * u
            du = fixed_mul(D_val, u_val)
            y_with_D = y_scan + du
            
            # 6. Gating: y = y * SiLU(gate)
            g_act = silu_fixed_approx(gate)
            y_out = fixed_mul(y_with_D, g_act)
            
            # Clamp output
            if y_out > MAX_INT: y_out = MAX_INT
            if y_out < MIN_INT: y_out = MIN_INT
            
            Y_final_list.append(y_out)
            
    # Save Output
    with open(OUT_Y_GOLD, 'w') as f:
        f.write('\n'.join([to_hex(v) for v in Y_final_list]))
        
    print(f"DONE! Generated {OUT_Y_GOLD} (Shape: {len(Y_final_list)})")
    print(f"Generated {OUT_A_HEX} and {OUT_D_HEX}")

if __name__ == "__main__":
    run()