import numpy as np
import math
import re

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
SCALE   = 4096.0 # 1 << 12

# --- ĐƯỜNG DẪN FILE ---
DIR_VEC    = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"
DIR_CPP    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/"

# Input Raw
FILE_A_LOG = DIR_VEC + "A_log.txt"

# Input Hex (Fixed-point -> Cần chia Scale để về Float)
FILE_X     = "conv_y_golden_ptb.txt"      # Channel-First (128, 1000)
FILE_B     = "scan_real_B_shared.txt"     # Token-First (1000, 16)
FILE_C     = "scan_real_C_shared.txt"     # Token-First (1000, 16)
FILE_DELTA = "gold_delta_final.txt"       # Token-First (1000, 128)

# Output Reference (PTB Raw Output)
FILE_PTB_RAW = DIR_CPP + "1013_13_Mixer_scan_output_raw.txt" # Shape (128, 1000)

# ==============================================================================
# HELPER
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def hex_to_float(hex_str):
    return to_signed(hex_str) / SCALE

def load_data():
    print("--- Loading Data ---")
    
    # 1. A_log -> A (Raw Float)
    A_log = np.zeros((D_INNER, D_STATE))
    try:
        with open(FILE_A_LOG) as f:
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    A_log[r, c] = v
    except: pass
    A = -np.exp(A_log) # Shape (128, 16)
    
    # --- MÔ PHỎNG HARDWARE CLAMPING (Q3.12) ---
    # Ép A về trong khoảng [-8, 7.99]
    print("!!! WARNING: Clamping A to Q3.12 range [-8.0, 7.99] !!!")
    A = np.clip(A, -8.0, 7.999) 

    # 2. X (Channel-First: 128, 1000)
    with open(FILE_X) as f: x_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    X = np.array(x_dat).reshape(D_INNER, SEQ_LEN)
    
    # 3. Delta (Token-First: 1000, 128) -> Convert to (1000, 128)
    with open(FILE_DELTA) as f: d_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    Delta = np.array(d_dat).reshape(SEQ_LEN, D_INNER)
    
    # 4. B, C (Token-First: 1000, 16)
    with open(FILE_B) as f: b_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    B = np.array(b_dat).reshape(SEQ_LEN, D_STATE)
    
    with open(FILE_C) as f: c_dat = [hex_to_float(l.strip()) for l in f if l.strip()]
    C = np.array(c_dat).reshape(SEQ_LEN, D_STATE)

    return A, X, Delta, B, C

# ==============================================================================
# SIMULATION
# ==============================================================================
def run():
    A, X, Delta, B, C = load_data()
    
    print(f"X range: {np.min(X):.4f} to {np.max(X):.4f}")
    print(f"Delta range: {np.min(Delta):.4f} to {np.max(Delta):.4f}")
    print(f"A range: {np.min(A):.4f} to {np.max(A):.4f}")
    
    scan_output_raw = np.zeros((D_INNER, SEQ_LEN))
    
    print("--- Simulating Scan Core (Pure Float Logic) ---")
    
    for d in range(D_INNER):
        h = np.zeros(D_STATE)
        A_d = A[d] # Vector (16,)
        
        for t in range(SEQ_LEN):
            # Inputs at step t
            u = X[d, t]          # Scalar
            delta = Delta[t, d]  # Scalar (Delta của channel d tại thời điểm t)
            
            B_t = B[t]           # Vector (16,)
            C_t = C[t]           # Vector (16,)
            
            # --- SSM Logic ---
            # discrete_A = exp(delta * A)
            disc_A = np.exp(delta * A_d)
            
            # discrete_B = (delta * B) * u
            disc_B_u = (delta * B_t) * u
            
            # Update h
            h = disc_A * h + disc_B_u
            
            # Compute y
            y = np.sum(h * C_t)
            
            scan_output_raw[d, t] = y

    # --- COMPARE ---
    print("--- Comparing with PTB Raw Output ---")
    try:
        ptb_ref = np.loadtxt(FILE_PTB_RAW) # Shape (128, 1000)
        
        # Check shape match
        if ptb_ref.shape != scan_output_raw.shape:
            print(f"Shape Mismatch! PTB: {ptb_ref.shape}, Calc: {scan_output_raw.shape}")
            if ptb_ref.shape == (SEQ_LEN, D_INNER): ptb_ref = ptb_ref.T
            
        diff = np.abs(scan_output_raw - ptb_ref)
        print(f"MAX Diff: {np.max(diff):.6f}")
        print(f"AVG Diff: {np.mean(diff):.6f}")
        
        print("\nSample D=0 (First 5):")
        print(f"Calc: {scan_output_raw[0, :5]}")
        print(f"PTB : {ptb_ref[0, :5]}")
        
        # Debug chi tiết tại bước đầu tiên
        print("\n--- DEBUG STEP 0 (Channel 0) ---")
        delta0 = Delta[0, 0]
        u0 = X[0, 0]
        A0 = A[0]
        B0 = B[0]
        C0 = C[0]
        disc_A0 = np.exp(delta0 * A0)
        disc_B0 = (delta0 * B0) * u0
        h0 = disc_B0 # Vì h_-1 = 0
        y0 = np.sum(h0 * C0)
        
        print(f"Delta[0,0]: {delta0:.6f}")
        print(f"X[0,0]    : {u0:.6f}")
        print(f"y[0,0] Calc: {y0:.6f}")
        
    except Exception as e:
        print(f"Error comparing: {e}")

if __name__ == "__main__":
    run()