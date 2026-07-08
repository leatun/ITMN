import numpy as np
import math

# ==============================================================================
# 1. CẤU HÌNH PHẦN CỨNG
# ==============================================================================
SEQ_LEN = 1000
D_MODEL = 64
D_INNER = 128
D_STATE = 16
DT_RANK = 4
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS # 4096

MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN ---
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/"
DIR_VEC    = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"
DIR_CPP    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/"

# --- FILES INPUT (Nạp vào TB) ---
FILE_X_IN        = "linear_x_input.txt"
FILE_W_L1        = "lin_real_w1_reordered.txt"
FILE_W_L2        = "lin_real_w2_reordered.txt"
FILE_W_CONV      = "conv_w_input_real.txt"
FILE_B_CONV      = "conv_b_input_real.txt"
FILE_W_XPROJ     = "w_xproj_reordered.txt"
FILE_W_DTPROJ    = "w_dt_proj_reordered.txt"
FILE_B_DTPROJ    = "b_dt_proj.txt"
FILE_A           = "scan_A_ptb.txt"
FILE_D           = "scan_D_ptb.txt"

# --- FILES GOLDEN (Để so sánh) ---
FILE_GOLD_PRIM   = "linear1_golden.txt"
FILE_GOLD_GATE   = "linear2_golden.txt"
FILE_GOLD_CONV   = "conv_y_golden_real.txt"
FILE_GOLD_B      = "scan_real_B_shared.txt"
FILE_GOLD_C      = "scan_real_C_shared.txt"
FILE_GOLD_DT     = "gold_dt_raw_hw.txt"
FILE_GOLD_DELTA  = "gold_delta_final.txt"
FILE_GOLD_SCAN   = "gold_scan_final.txt"

# ==============================================================================
# 2. BỘ VI XỬ LÝ "NHÀ NGHÈO" (HARDWARE EMULATOR)
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    """Kẹp 16-bit Signed"""
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def hw_mul(a, b):
    """Nhân -> Dịch 12 -> Kẹp"""
    prod = int(a) * int(b)
    res = prod >> FRAC_BITS # Arithmetic shift
    return sat16(res)

def hw_mac_vec(vec_in, vec_w, bias=0):
    """Mô phỏng PE tích lũy"""
    acc = 0
    for i in range(len(vec_in)):
        term = (int(vec_in[i]) * int(vec_w[i])) >> FRAC_BITS
        acc = sat16(acc + term) # Kẹp ngay sau mỗi lần cộng (Logic Unified_PE)
    
    # Cộng bias
    if bias != 0:
        acc = sat16(acc + bias)
    return acc

def hw_silu(x):
    """Mô phỏng SiLU PWL hoặc Float Approx"""
    # Dùng float approx để nhanh, sai số +/- 1 unit chấp nhận được
    x_real = x / SCALE
    res = x_real / (1.0 + math.exp(-x_real))
    return sat16(int(round(res * SCALE)))

def hw_softplus(x):
    """Mô phỏng Softplus"""
    x_real = x / SCALE
    try:
        res = math.log(1.0 + math.exp(x_real))
    except OverflowError:
        res = x_real # Khi x lớn, softplus(x) approx x
    return sat16(int(round(res * SCALE)))

def hw_exp(x):
    """Mô phỏng Exp Unit"""
    x_real = x / SCALE
    try:
        res = math.exp(x_real)
    except OverflowError:
        res = 8.0 # Max range Q3.12 roughly
    
    val = int(round(res * SCALE))
    if val > MAX_INT: return MAX_INT # 7.999
    if val < 0: return 0 
    return val

# ==============================================================================
# 3. DATA LOADING (RECONSTRUCT MATRICES)
# ==============================================================================
def load_data():
    print("--- Loading Inputs & Weights ---")
    
    # Input X (1000, 64)
    with open(FILE_X_IN) as f: 
        x_flat = [to_signed(l.strip()) for l in f if l.strip()]
    X_in = np.array(x_flat).reshape(SEQ_LEN, D_MODEL)

    # Helper load weights (Cần biết cấu trúc reorder của file text)
    # Giả sử file Reorder lưu: Chunk -> Col -> Row (như script tạo file đã làm)
    def load_weight_matrix(fname, rows, cols):
        with open(fname) as f: w_flat = [to_signed(l.strip()) for l in f if l.strip()]
        # Reconstruct logic ngược lại của script tạo file:
        # File: 1 dòng hex = 1 weight.
        # Thứ tự: Chunk 0 (Rows 0-15) -> Col 0..Cols-1 -> Chunk 1...
        W = np.zeros((rows, cols), dtype=int)
        idx = 0
        num_chunks = (rows + 15) // 16
        for chunk in range(num_chunks):
            start_row = chunk * 16
            for c in range(cols):
                for r_sub in range(16):
                    r = start_row + r_sub
                    if r < rows and idx < len(w_flat):
                        W[r, c] = w_flat[idx]
                    idx += 1
        return W

    W1 = load_weight_matrix(FILE_W_L1, D_INNER, D_MODEL)
    W2 = load_weight_matrix(FILE_W_L2, D_INNER, D_MODEL)
    
    # Conv Weight (128, 4)
    # File Conv thường lưu: Channel 0 (4 taps)... Channel 127
    with open(FILE_W_CONV) as f: w_conv_flat = [to_signed(l.strip()) for l in f]
    W_Conv = np.array(w_conv_flat).reshape(D_INNER, 4)
    
    with open(FILE_B_CONV) as f: b_conv_flat = [to_signed(l.strip()) for l in f]
    B_Conv = np.array(b_conv_flat) # (128,)

    # X_Proj Weights (48, 128) -> B, C, dt
    W_XProj = load_weight_matrix(FILE_W_XPROJ, 36, D_INNER) # Thực tế file có padding 48, lấy 36
    W_B = W_XProj[0:16]
    W_C = W_XProj[16:32]
    W_dt_raw = W_XProj[32:36]

    # DT Proj Weights (128, 4)
    W_DTProj = load_weight_matrix(FILE_W_DTPROJ, D_INNER, DT_RANK)
    with open(FILE_B_DTPROJ) as f: b_dt_flat = [to_signed(l.strip()) for l in f]
    B_DT = np.array(b_dt_flat)

    # Static A, D
    with open(FILE_A) as f: a_flat = [to_signed(l.strip()) for l in f]
    A = np.array(a_flat).reshape(D_INNER, D_STATE)
    
    with open(FILE_D) as f: d_flat = [to_signed(l.strip()) for l in f]
    D = np.array(d_flat)

    return X_in, W1, W2, W_Conv, B_Conv, W_B, W_C, W_dt_raw, W_DTProj, B_DT, A, D

# ==============================================================================
# 4. SIMULATION FULL FLOW
# ==============================================================================
def compare(name, hw_calc, gold_file, shape):
    print(f"\n--- Checking {name} ---")
    with open(gold_file) as f: gold_flat = [to_signed(l.strip()) for l in f if l.strip()]
    gold = np.array(gold_flat).reshape(shape)
    
    diff = np.abs(hw_calc - gold)
    print(f"Max Diff: {np.max(diff)}")
    print(f"Avg Diff: {np.mean(diff):.2f}")
    if np.max(diff) > 100: print(f"-> {name} has LARGE ERROR accumulation!")
    else: print(f"-> {name} looks GOOD.")

def run():
    X_in, W1, W2, W_Conv, B_Conv, W_B, W_C, W_dt, W_DT2, B_DT, A, D = load_data()
    
    # --- PHASE 1: LINEAR ---
    print("\nRunning Phase 1 (Linear)...")
    X_Prime = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    Gate    = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    
    for t in range(SEQ_LEN):
        for ch in range(D_INNER):
            X_Prime[t, ch] = hw_mac_vec(X_in[t], W1[ch])
            Gate[t, ch]    = hw_mac_vec(X_in[t], W2[ch])
            
    compare("Phase 1 X_Prime", X_Prime, FILE_GOLD_PRIM, (SEQ_LEN, D_INNER))
    compare("Phase 1 Gate", Gate, FILE_GOLD_GATE, (SEQ_LEN, D_INNER))

    # --- PHASE 2: CONV1D ---
    print("\nRunning Phase 2 (Conv1D)...")
    X_Conv_Out = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    # Transpose X_Prime to (Channel, Time) for easier Conv
    X_Prime_T = X_Prime.T 
    
    for ch in range(D_INNER):
        # Shift Reg logic
        hist = [0, 0, 0, 0] # 4 taps
        w_knl = W_Conv[ch]
        bias = B_Conv[ch]
        
        for t in range(SEQ_LEN):
            # Shift in new data
            hist = [X_Prime_T[ch, t]] + hist[:-1] 
            # Conv MAC
            # Hardware: hist[0]*w[0] + hist[1]*w[1]...
            # Note: Kernel order matters. Assuming W_Conv is [tap0, tap1, tap2, tap3]
            acc = 0
            for k in range(4):
                acc = sat16(acc + ((hist[k] * w_knl[k]) >> FRAC_BITS))
            
            # Add Bias
            acc = sat16(acc + bias)
            # SiLU
            X_Conv_Out[t, ch] = hw_silu(acc)
            
    compare("Phase 2 Conv", X_Conv_Out, FILE_GOLD_CONV, (SEQ_LEN, D_INNER)) # Note: Gold file shape logic

    # --- PHASE 3: PROJECTIONS ---
    print("\nRunning Phase 3 (X_Proj & DT_Proj)...")
    B_hw = np.zeros((SEQ_LEN, D_STATE), dtype=int)
    C_hw = np.zeros((SEQ_LEN, D_STATE), dtype=int)
    DT_Raw_hw = np.zeros((SEQ_LEN, 4), dtype=int)
    Delta_hw  = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    
    for t in range(SEQ_LEN):
        x = X_Conv_Out[t]
        # Calc B
        for n in range(D_STATE): B_hw[t, n] = hw_mac_vec(x, W_B[n])
        # Calc C
        for n in range(D_STATE): C_hw[t, n] = hw_mac_vec(x, W_C[n])
        # Calc DT Raw
        for r in range(4):       DT_Raw_hw[t, r] = hw_mac_vec(x, W_dt[r])
        
        # Phase 3.2: Calc Delta
        for ch in range(D_INNER):
            # Linear DT Proj
            dt_lin = hw_mac_vec(DT_Raw_hw[t], W_DT2[ch], B_DT[ch])
            # Softplus
            Delta_hw[t, ch] = hw_softplus(dt_lin)

    compare("Phase 3 B", B_hw, FILE_GOLD_B, (SEQ_LEN, D_STATE))
    compare("Phase 3 C", C_hw, FILE_GOLD_C, (SEQ_LEN, D_STATE))
    compare("Phase 3 Delta", Delta_hw, FILE_GOLD_DELTA, (SEQ_LEN, D_INNER))

    # --- PHASE 4: SCAN CORE ---
    print("\nRunning Phase 4 (Scan Core) with ACCUMULATED ERROR...")
    Y_Scan = np.zeros((SEQ_LEN, D_INNER), dtype=int) # Token First for convenience
    
    for d in range(D_INNER):
        h = np.zeros(D_STATE, dtype=int)
        A_row = A[d]
        D_val = D[d]
        
        for t in range(SEQ_LEN):
            u = X_Conv_Out[t, d] # Input chứa sai số từ Phase 2
            delta = Delta_hw[t, d] # Delta chứa sai số từ Phase 3
            gate = Gate[t, d]      # Gate chứa sai số từ Phase 1
            
            # B, C chứa sai số từ Phase 3
            B_t = B_hw[t]
            C_t = C_hw[t]
            
            # --- SSM HW Logic ---
            disc_A = np.zeros(D_STATE, dtype=int)
            dBx = np.zeros(D_STATE, dtype=int)
            
            for n in range(D_STATE):
                # A_bar = exp(delta * A)
                val_A = hw_mul(delta, A_row[n])
                disc_A[n] = hw_exp(val_A)
                
                # B_bar = delta * B
                val_B = hw_mul(delta, B_t[n])
                # dBx = B_bar * u
                dBx[n] = hw_mul(val_B, u)
                
                # Update H
                # h = sat( sat(A_bar * h) + dBx )
                term1 = hw_mul(disc_A[n], h[n])
                h[n] = sat16(term1 + dBx[n])
                
            # Output y = C * h
            y_scan = 0
            for n in range(D_STATE):
                term = hw_mul(C_t[n], h[n])
                y_scan = sat16(y_scan + term)
                
            # Skip D: y = y + D*u
            du = hw_mul(D_val, u)
            y_wd = sat16(y_scan + du)
            
            # Gating: y = y * silu(gate)
            g_act = hw_silu(gate)
            Y_Scan[t, d] = hw_mul(y_wd, g_act)

    # So sánh với file Golden Scan (được tạo từ Python Float/PTB)
    print("\n--- FINAL VERDICT (Phase 4 Output) ---")
    # Load Golden Channel-First -> Reshape -> Transpose to Token-First for comparison
    with open(FILE_GOLD_SCAN) as f: gold_s = [to_signed(l.strip()) for l in f if l.strip()]
    Gold_Scan = np.array(gold_s).reshape(D_INNER, SEQ_LEN).T 
    
    diff = np.abs(Y_Scan - Gold_Scan)
    print(f"MAX Diff Final: {np.max(diff)}")
    print(f"AVG Diff Final: {np.mean(diff):.2f}")
    
    # Phân tích kênh 49 (nơi cậu thấy lỗi)
    print("\nDeep Dive Channel 49:")
    print(f"HW (T0..9): {Y_Scan[:10, 49]}")
    print(f"GD (T0..9): {Gold_Scan[:10, 49]}")
    print(f"Diff: {diff[:10, 49]}")

if __name__ == "__main__":
    run()