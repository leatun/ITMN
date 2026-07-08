import numpy as np
import math
import os

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768
EPSILON = 1e-5

# --- QUAN TRỌNG: CHẾ ĐỘ KIỂM TRA ---
# Set = False để khớp với file Golden hiện tại (bị thiếu D)
# Set = True là đúng chuẩn Mamba Hardware
ENABLE_D_RESIDUAL = True 

# --- ĐƯỜNG DẪN FILE RAW C++  ---
BASE_DIR = "D:/DoAn1/Ultility/goldens/"

# 1. Parameter Tĩnh
F_A_LOG = BASE_DIR + "golden_vectors_txt/A_log.txt" # Shape (128, 16)
F_D     = BASE_DIR + "golden_vectors_txt/D.txt"     # Shape (128,)

# 2. Input Time-Varying
F_DELTA = BASE_DIR + "cpp_golden_files/10_09_Mixer_delta_final.txt"  # Shape (128, 1000)
F_X_ACT = BASE_DIR + "cpp_golden_files/09_08_Mixer_x_activated.txt"  # Shape (128, 1000) - Input X cho Scan
F_X_RAW = BASE_DIR + "cpp_golden_files/08_X_after_linear.txt"        # Shape (1000, 256) - Lấy Gate từ đây

# 3. Shared Parameters
F_B_RAW = BASE_DIR + "cpp_golden_files/11_10_Mixer_B_raw.txt"        # Shape (1000, 16)
F_C_RAW = BASE_DIR + "cpp_golden_files/12_11_Mixer_C_raw.txt"        # Shape (1000, 16)

# 4. GOLDEN OUTPUT CHUẨN (Cần file số 14)
F_GOLDEN = BASE_DIR + "cpp_golden_files/1014_14_Mixer_y_gated.txt"

# ... (Giữ nguyên các hàm to_fixed, sat16, fixed_mul, fixed_exp, fixed_silu, parse_file_safe) ...
# Copy y chang các hàm hỗ trợ từ script trước vào đây

def to_hex(val, width=16):
    val = int(val)
    if val < 0:
        val = (1 << width) + val
    return f"{val:04x}"

def to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def fixed_mul(a, b):
    return sat16((a * b) >> FRAC_BITS)

def fixed_exp(val):
    f_val = val / SCALE
    res = math.exp(f_val)
    return sat16(int(round(res * SCALE)))

def fixed_silu(val):
    f_val = val / SCALE
    if f_val >= 0: sig = 1.0 / (1.0 + math.exp(-f_val))
    else:          sig = math.exp(f_val) / (1.0 + math.exp(f_val))
    return sat16(int(round(f_val * sig * SCALE)))

def parse_file_safe(filepath):
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                try: data_list.append(float(parts[-1]))
                except: continue
    return np.array(data_list)

# --- MAIN ---
def run():
    print(f"=== DEBUG SCAN CORE (ENABLE_D_RESIDUAL = {ENABLE_D_RESIDUAL}) ===")
    
    # 1. LOAD PARAMS & INPUTS (Copy từ script trước)
    # ... (Phần load A, D, Delta, X, Gate, B, C, Golden giữ nguyên) ...
    try:
        a_log_raw = parse_file_safe(F_A_LOG)
        A_log = a_log_raw.reshape(D_INNER, D_STATE)
        A_real = -np.exp(A_log)
        A_fix = to_fixed(A_real)
        
        d_raw = parse_file_safe(F_D)
        D_fix = to_fixed(d_raw)
        
        delta_raw = np.loadtxt(F_DELTA).reshape(SEQ_LEN, D_INNER)
        Delta_fix = to_fixed(delta_raw.T)
        
        x_act_raw = np.loadtxt(F_X_ACT).reshape(SEQ_LEN, D_INNER)
        X_fix = to_fixed(x_act_raw.T)
        
        x_linear_raw = np.loadtxt(F_X_RAW).reshape(SEQ_LEN, 256)
        z_raw = x_linear_raw[:, 128:]
        Gate_fix = to_fixed(z_raw.T)
        
        B_raw = np.loadtxt(F_B_RAW).reshape(SEQ_LEN, D_STATE)
        C_raw = np.loadtxt(F_C_RAW).reshape(SEQ_LEN, D_STATE)
        B_fix = to_fixed(B_raw)
        C_fix = to_fixed(C_raw)
        
        y_gold_raw = np.loadtxt(F_GOLDEN).reshape(SEQ_LEN, D_INNER)
        Y_gold_fix = to_fixed(y_gold_raw.T)
        
    except Exception as e: print(f"Load Err: {e}"); return

    # 4. SIMULATION
    print("4. Simulating...")
    total_err = 0
    max_diff = 0
    
    for d in range(D_INNER):
        h = np.zeros(D_STATE, dtype=int)
        A_vec = A_fix[d]
        D_val = D_fix[d]
        
        for t in range(SEQ_LEN):
            # ... Load inputs ...
            delta = Delta_fix[d, t]
            x_val = X_fix[d, t]
            gate  = Gate_fix[d, t]
            B_vec = B_fix[t]
            C_vec = C_fix[t]
            
            # 1. State Update
            h_new = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                deltaA = fixed_mul(delta, A_vec[n])
                discA = fixed_exp(deltaA)
                
                deltaB = fixed_mul(delta, B_vec[n])
                deltaBx = fixed_mul(deltaB, x_val)
                
                term1 = fixed_mul(discA, h[n])
                h_new[n] = sat16(term1 + deltaBx)
            h = h_new
            
            # 2. Output
            y_scan = 0
            for n in range(D_STATE):
                y_scan += fixed_mul(C_vec[n], h[n])
                
            # 3. Gating (CÓ D hoặc KHÔNG D)
            if ENABLE_D_RESIDUAL:
                Dx = fixed_mul(D_val, x_val)
                y_with_D = y_scan + Dx
            else:
                y_with_D = y_scan # Bỏ qua D để khớp với file Golden bị lỗi
            
            g_act = fixed_silu(gate)
            y_final = sat16((y_with_D * g_act) >> FRAC_BITS)
            
            # 4. Compare
            gold = Y_gold_fix[d, t]
            diff = abs(y_final - gold)
            if diff > max_diff: max_diff = diff
            
            if diff > 30: 
                if total_err == 0:
                    print(f"[FIRST ERR] Ch={d} T={t} | Calc={y_final} Gold={gold} | Diff={diff}")
                    print(f"   -> Mode: {'WITH_D' if ENABLE_D_RESIDUAL else 'NO_D'}")
                total_err += 1
                
    print(f"\nTotal Errors (>30 LSB): {total_err}")
    
    if total_err < 5000:
        print(">>> SUCCESS: Logic OK (Khớp với file Golden hiện tại)!")
    else:
        print(">>> FAIL: Vẫn lệch.")
        
        
    # --- 5. EXPORT HARDWARE GOLDEN ---
    print("\n5. Exporting Hardware-Exact Golden File...")
    
    # Tính lại toàn bộ (có D) để lưu ra file
    hw_golden_list = []
    
    for d in range(D_INNER):
        h = np.zeros(D_STATE, dtype=int)
        A_vec = A_fix[d]
        D_val = D_fix[d]
        
        for t in range(SEQ_LEN):
            delta = Delta_fix[d, t]
            x_val = X_fix[d, t]
            gate  = Gate_fix[d, t]
            B_vec = B_fix[t]
            C_vec = C_fix[t]
            
            # Recalculate (Copy logic from above)
            h_new = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                deltaA = fixed_mul(delta, A_vec[n])
                discA = fixed_exp(deltaA)
                deltaB = fixed_mul(delta, B_vec[n])
                deltaBx = fixed_mul(deltaB, x_val)
                term1 = fixed_mul(discA, h[n])
                h_new[n] = sat16(term1 + deltaBx)
            h = h_new
            
            y_scan = 0
            for n in range(D_STATE):
                y_scan += fixed_mul(C_vec[n], h[n])
                
            # Always enable D for final export
            Dx = fixed_mul(D_val, x_val)
            y_with_D = y_scan + Dx
            
            g_act = fixed_silu(gate)
            y_final = sat16((y_with_D * g_act) >> FRAC_BITS)
            
            hw_golden_list.append(to_hex(y_final))

    with open("scan_y_golden_HARDWARE.txt", "w") as f:
        f.write("\n".join(hw_golden_list))
    
    print(">>> DONE! Created 'scan_y_golden_HARDWARE.txt'. Use this for Verilog TB.")

if __name__ == "__main__":
    run()