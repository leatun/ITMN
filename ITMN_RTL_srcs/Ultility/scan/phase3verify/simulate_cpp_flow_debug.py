import numpy as np
import math
import re

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

# --- ĐƯỜNG DẪN FILE ---
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/"
DIR_VEC    = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"
DIR_CPP    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/"

# Input Raw (Float)
FILE_A_LOG = DIR_VEC + "A_log.txt"
FILE_D     = DIR_VEC + "D.txt"

# Input Hex (Fixed-point từ Hardware Flow) -> Cần convert về Float để mô phỏng C++
FILE_X     = "conv_y_golden_ptb.txt"      # Channel-First
FILE_GATE  = "linear2_golden.txt"         # Token-First
FILE_B     = "scan_real_B_shared.txt"     # Token-First
FILE_C     = "scan_real_C_shared.txt"     # Token-First
FILE_DELTA = "gold_delta_final.txt"       # Token-First

# Output Reference (PTB)
FILE_PTB_TARGET = DIR_CPP + "1014_14_Mixer_y_gated.txt"

# ==============================================================================
# HELPER LOADER
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def hex_to_float(hex_str):
    return to_signed(hex_str) / SCALE

def load_data():
    print("--- Loading Data & Converting to Float ---")
    
    # 1. A_log (Raw Float) -> A_cpp
    A_log = np.zeros((D_INNER, D_STATE))
    try:
        with open(FILE_A_LOG) as f:
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    A_log[r, c] = v
    except: pass
    # C++: A_cpp[i][j] = -std::exp(weights->A_log...);
    A = -np.exp(A_log)

    # 2. D (Raw Float)
    D = np.zeros(D_INNER)
    try:
        with open(FILE_D) as f:
            idx = 0
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 2: D[int(parts[0])] = float(parts[1])
                elif len(parts) == 1: 
                    D[idx] = float(parts[0]); idx += 1
    except: pass

    # 3. Hex Files -> Float
    with open(FILE_X) as f: x_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    X = np.array(x_dat).reshape(D_INNER, SEQ_LEN) # [D, L]
    
    with open(FILE_GATE) as f: g_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    Gate = np.array(g_dat).reshape(SEQ_LEN, D_INNER).T # [L, D] -> Transpose -> [D, L]
    
    with open(FILE_DELTA) as f: d_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    Delta = np.array(d_dat).reshape(SEQ_LEN, D_INNER).T # [L, D] -> Transpose -> [D, L] (Khớp logic C++)
    
    with open(FILE_B) as f: b_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    B = np.array(b_dat).reshape(SEQ_LEN, D_STATE) # [L, N]
    
    with open(FILE_C) as f: c_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    C = np.array(c_dat).reshape(SEQ_LEN, D_STATE) # [L, N]

    return A, D, X, Gate, Delta, B, C

# ==============================================================================
# MAIN.CPP FLOW SIMULATION
# ==============================================================================
def run_cpp_logic():
    A, D, X, Gate, Delta, B, C = load_data()
    
    print("--- Simulating main.cpp Logic (Float64) ---")
    
    # Init Outputs
    # scan_output_raw[D_INNER][SEQ_LEN]
    scan_output_raw = np.zeros((D_INNER, SEQ_LEN))
    
    # 1. Pre-calculate Discrete Parameters (Loop Fusion in C++ logic)
    # discrete_A[d][l][n] = exp(A[d][n] * delta[d][l])
    # deltaB_u[d][l][n]   = (delta[d][l] * B[l][n]) * x[d][l]
    
    print("Step 1: Discretization...")
    # Sử dụng numpy broadcasting để mô phỏng loop cho nhanh và chính xác
    # A: [D, N], Delta: [D, L] -> einsum -> [D, L, N]
    # A[d, n] * Delta[d, l]
    # Reshape A: [D, 1, N], Delta: [D, L, 1]
    
    A_exp = A[:, np.newaxis, :]        # [D, 1, N]
    Delta_exp = Delta[:, :, np.newaxis] # [D, L, 1]
    
    discrete_A = np.exp(A_exp * Delta_exp) # [D, L, N]
    
    # DeltaB_u calculation
    # B: [L, N] -> [1, L, N]
    # X: [D, L] -> [D, L, 1]
    B_exp = B[np.newaxis, :, :] # [1, L, N]
    X_exp = X[:, :, np.newaxis] # [D, L, 1]
    
    # (Delta * B) * X
    # Delta [D, L, 1] * B [1, L, N] -> [D, L, N]
    deltaB = Delta_exp * B_exp 
    deltaB_u = deltaB * X_exp      # [D, L, N]
    
    print("Step 2: Scan Core Recurrence...")
    # scan_core(discrete_A, deltaB_u, C_raw, scan_output_raw)
    
    for d in range(D_INNER):
        h = np.zeros(D_STATE)
        for t in range(SEQ_LEN):
            # h_t = A_bar * h_{t-1} + B_bar_u
            # discrete_A[d, t, :] vector N
            h = discrete_A[d, t, :] * h + deltaB_u[d, t, :]
            
            # y_t = C_t * h_t
            # C[t, :] vector N
            y_val = np.sum(C[t, :] * h)
            scan_output_raw[d, t] = y_val

    print("Step 3: Post-processing (Skip Connection + Gating)...")
    
    # scan_output_with_D = scan_output + D * x
    # D: [D], X: [D, L]
    D_exp = D[:, np.newaxis]
    scan_output_with_D = scan_output_raw + (D_exp * X)
    
    # z_activated = silu(gate)
    # Gate đã load: [D, L] (đã transpose)
    # Hàm silu chuẩn: x * sigmoid(x)
    z_activated = Gate * (1.0 / (1.0 + np.exp(-Gate)))
    
    # y_gated = y_rearranged * z_activated
    y_gated = scan_output_with_D * z_activated
    
    return y_gated

# ==============================================================================
# VERIFY WITH PTB
# ==============================================================================
def compare_with_ptb(calc_y):
    print("--- Comparing with PTB Reference ---")
    try:
        # Load PTB (128, 1000)
        ptb_ref = np.loadtxt(FILE_PTB_TARGET)
        # Check shape
        if ptb_ref.shape == (SEQ_LEN, D_INNER): ptb_ref = ptb_ref.T
        
        diff = np.abs(calc_y - ptb_ref)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"MAX Diff (Float vs Float): {max_diff:.6f}")
        print(f"AVG Diff: {avg_diff:.6f}")
        
        print("\nSample D=0, T=0..9:")
        print(f"Cpp Sim: {calc_y[0, :10]}")
        print(f"PTB Ref: {ptb_ref[0, :10]}")
        
        if max_diff < 0.1:
            print("\n>>> SUCCESS: C++ Logic Matches PTB! (Hardware Logic is correct) <<<")
            print("Vấn đề nằm ở Fixed-Point Precision (Exp/Mul/Add).")
        else:
            print("\n>>> FAILURE: C++ Logic Mismatch! <<<")
            print("Có thể sai ở: Thứ tự kênh B/C/Delta hoặc cách reshape X/Gate.")
            
    except Exception as e:
        print(f"Cannot compare: {e}")

if __name__ == "__main__":
    y_out = run_cpp_logic()
    compare_with_ptb(y_out)