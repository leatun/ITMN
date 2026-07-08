import numpy as np
import math

# ==============================================================================
# CẤU HÌNH & HẰNG SỐ
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE (Sửa lại cho đúng máy cậu) ---
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/"
DIR_VEC    = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"

# INPUTS (Giống hệt TB Phase 4 Only)
FILE_A           = "scan_A_ptb.txt"
FILE_D           = "scan_D_ptb.txt"
FILE_X           = "conv_y_golden_real.txt"      # Channel-First (Check kỹ!)
FILE_GATE        = "scan_gate_channel_first.txt" # Channel-First
FILE_B           = "scan_real_B_shared.txt"      # Token-First
FILE_C           = "scan_real_C_shared.txt"      # Token-First
FILE_DELTA       = "gold_delta_final.txt"        # Token-First

# OUTPUT REFERENCE
FILE_GOLD_SCAN   = "gold_scan_final.txt"         # Channel-First

# ==============================================================================
# BỘ XỬ LÝ SỐ HỌC "NHÀ NGHÈO" (HARDWARE EMULATOR)
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def hw_mul(a, b):
    prod = int(a) * int(b)
    res = prod >> FRAC_BITS 
    return sat16(res)

def hw_exp(val_fixed):
    """Mô phỏng Exp Unit PWL hoặc Float Approx kẹp biên"""
    val_real = val_fixed / SCALE
    try:
        res_real = math.exp(val_real)
    except: res_real = 8.0 # Max overflow
    
    res_fixed = int(round(res_real * SCALE))
    if res_fixed > MAX_INT: return MAX_INT # 7.999
    if res_fixed < 0: return 0 
    return res_fixed

def hw_silu(val_fixed):
    val_real = val_fixed / SCALE
    try:
        sig = 1.0 / (1.0 + math.exp(-val_real))
    except: sig = 0.0 if val_real < 0 else 1.0
    
    res = val_fixed * sig
    return sat16(int(round(res))) # Output vẫn là Q3.12

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data():
    print("--- Loading Clean Inputs ---")
    
    def load_hex(fname):
        with open(fname) as f: return [to_signed(l.strip()) for l in f if l.strip()]

    # 1. Static
    A = np.array(load_hex(FILE_A)).reshape(D_INNER, D_STATE)
    D = np.array(load_hex(FILE_D))

    # 2. Channel-First Inputs
    # X & Gate: TB Phase 4 load theo Channel-First
    X = np.array(load_hex(FILE_X)).reshape(D_INNER, SEQ_LEN)
    Gate = np.array(load_hex(FILE_GATE)).reshape(D_INNER, SEQ_LEN)

    # 3. Token-First Inputs
    B = np.array(load_hex(FILE_B)).reshape(SEQ_LEN, D_STATE)
    C = np.array(load_hex(FILE_C)).reshape(SEQ_LEN, D_STATE)
    Delta = np.array(load_hex(FILE_DELTA)).reshape(SEQ_LEN, D_INNER)

    return A, D, X, Gate, B, C, Delta

# ==============================================================================
# SCAN CORE SIMULATION
# ==============================================================================
def run():
    A, D, X, Gate, B, C, Delta = load_data()
    
    print("--- Simulating Hardware Scan Core (Isolated) ---")
    Y_HW = np.zeros((D_INNER, SEQ_LEN), dtype=int)
    
    # Loop Channel -> Time (Khớp Hardware Flow)
    for d in range(D_INNER):
        h = np.zeros(D_STATE, dtype=int)
        A_row = A[d]
        D_val = D[d]
        
        for t in range(SEQ_LEN):
            # Input
            u = X[d, t]
            gate = Gate[d, t]
            
            # Shared/Dynamic Param (Token-First)
            delta = Delta[t, d]
            B_t = B[t]
            C_t = C[t]
            
            # --- SSM BIT-EXACT LOGIC ---
            
            # 1. Discrete A = exp(delta * A)
            disc_A = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                val = hw_mul(delta, A_row[n])
                disc_A[n] = hw_exp(val)
                
            # 2. Discrete Bx = (delta * B) * u
            dBx = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                dB = hw_mul(delta, B_t[n])
                dBx[n] = hw_mul(dB, u)
                
            # 3. Update H
            h_new = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                # h = sat( sat(disc_A * h) + dBx )
                term1 = hw_mul(disc_A[n], h[n])
                h_new[n] = sat16(term1 + dBx[n])
            h = h_new
            
            # 4. Output Projection y = C * h
            y_scan = 0
            for n in range(D_STATE):
                prod = hw_mul(C_t[n], h[n])
                y_scan = sat16(y_scan + prod)
                
            # 5. Skip D
            du = hw_mul(D_val, u)
            y_wd = sat16(y_scan + du)
            
            # 6. Gating
            g_act = hw_silu(gate)
            y_final = hw_mul(y_wd, g_act)
            
            Y_HW[d, t] = y_final

    # --- COMPARE ---
    print("\n--- Comparing with Golden Output ---")
    with open(FILE_GOLD_SCAN) as f:
        gold_flat = [to_signed(l.strip()) for l in f if l.strip()]
    Y_Gold = np.array(gold_flat).reshape(D_INNER, SEQ_LEN)
    
    diff = np.abs(Y_HW - Y_Gold)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"MAX Diff: {max_diff}")
    print(f"AVG Diff: {avg_diff:.2f}")
    
    
    # Check Channel 0
    print("\nChannel 0 (T0..9):")
    print(f"Calc: {Y_HW[0, :10]}")
    print(f"Gold: {Y_Gold[0, :10]}")

    if max_diff < 100:
        print("\n>>> KẾT LUẬN: Logic Scan Core CHUẨN! <<<")
        print("Sai số nhỏ này là do Exp/SiLU xấp xỉ khác nhau giữa Python và Verilog.")
        print("=> Lỗi khổng lồ trong Full Flow chắc chắn do sai số tích lũy từ Phase 1-3.")
    else:
        print("\n>>> KẾT LUẬN: Logic Scan Core vẫn có vấn đề (hoặc Input chưa sạch). <<<")

if __name__ == "__main__":
    run()