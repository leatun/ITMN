import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
KERNEL_SIZE = 4
FRAC_BITS = 10
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE (CHECK LẠI NHA) ---
FILE_X_FLOAT    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/08_X_after_linear.txt"
FILE_Y_FLOAT    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/09_08_Mixer_x_activated.txt"
FILE_W_FLOAT    = "D:/DoAn1/Ultility/conv/V2/ptb/conv1d_weight.txt"
FILE_B_FLOAT    = "D:/DoAn1/Ultility/conv/V2/ptb/conv1d_bias.txt"

# --- HÀM HỖ TRỢ ---
def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def sigmoid_stable(x: float) -> float:
    if x >= 0: return 1.0 / (1.0 + np.exp(-x))
    else:      return np.exp(x) / (1.0 + np.exp(x))

def silu_fixed(x_fixed: int) -> int:
    x_real = int(x_fixed) / SCALE
    y_real = x_real * sigmoid_stable(x_real)
    val = np.round(y_real * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

# --- MAIN ---
def run():
    print("=== PREPARE CONV DATA (FINAL) ===")
    
    # 1. LOAD INPUT (CẮT 128 CỘT)
    try:
        
        x_raw = np.loadtxt(FILE_X_FLOAT)
        # Cắt lấy 128 kênh đầu tiên
        X_matrix = x_raw[:, :D_INNER] 
        X_fixed = float_to_fixed(X_matrix)
        print(f"1. Input Loaded & Sliced. Shape: {X_matrix.shape}")
    except Exception as e:
        print(f"Error loading Input: {e}"); return

    # 2. LOAD GOLDEN Y (C++)
    try:
        y_raw = np.loadtxt(FILE_Y_FLOAT)
        if y_raw.shape == (D_INNER, SEQ_LEN):
             Y_gold_matrix = y_raw.T
        else:
             Y_gold_matrix = y_raw
        Y_gold_fixed = float_to_fixed(Y_gold_matrix)
        print(f"2. Golden Y Loaded. Shape: {Y_gold_fixed.shape}")
    except:
        print("Warning: No Golden Y found."); Y_gold_fixed = None

    # 3. LOAD WEIGHT/BIAS (PARSE INDEX)
    # Load Weight
    W_list = []
    with open(FILE_W_FLOAT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0: W_list.append(float(parts[-1]))
    W_fixed = float_to_fixed(np.array(W_list).reshape(D_INNER, KERNEL_SIZE))
    
    # Load Bias
    b_raw = np.loadtxt(FILE_B_FLOAT)
    if len(b_raw.shape) > 1:
         b_list = []
         with open(FILE_B_FLOAT, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0: b_list.append(float(parts[-1]))
         b_fixed = float_to_fixed(np.array(b_list))
    else:
         b_fixed = float_to_fixed(b_raw)
    print("3. Weight/Bias Loaded.")

    # 4. SIMULATION (PYTHON) - DÙNG FLIPPED WEIGHT
    print("4. Simulating Hardware Logic (With Flipped Weights)...")
    Y_sim_fixed = np.zeros_like(X_fixed)
    
    for c in range(D_INNER):
        bias_val = b_fixed[c]
        weights = W_fixed[c]
        history = [0] * KERNEL_SIZE
        
        for t in range(SEQ_LEN):
            x_val = X_fixed[t, c]
            history.insert(0, x_val)
            history.pop()
            
            acc = bias_val
            for k in range(KERNEL_SIZE):
                # QUAN TRỌNG: NHÂN NGƯỢC (W3 trước, W0 sau)
                # KERNEL_SIZE - 1 - k
                w_idx = KERNEL_SIZE - 1 - k
                acc += (history[k] * weights[w_idx]) >> FRAC_BITS
            
            acc = max(min(acc, MAX_INT), MIN_INT)
            Y_sim_fixed[t, c] = silu_fixed(acc)

    # 5. SO SÁNH VỚI C++
    if Y_gold_fixed is not None:
        diff = np.abs(Y_sim_fixed - Y_gold_fixed)
        print(f"   Max Diff (PySim vs C++): {np.max(diff)} LSB")
        # Nếu nhỏ (<20) là OK

    # 6. XUẤT FILE HEX CHO VERILOG
    print("5. Exporting Hex Files...")
    
    def save_folded(fname, data_2d):
        with open(fname, 'w') as f:
            for g in range(D_INNER // 16):
                start_ch = g * 16
                end_ch = start_ch + 16
                for t in range(SEQ_LEN):
                    chunk = data_2d[t, start_ch:end_ch]
                    for val in chunk:
                        f.write(to_hex(val) + "\n")

    save_folded("conv_x_input_real_10F.txt", X_fixed)
    
    # Dùng kết quả Sim của Python làm Golden cho Verilog (để khớp 100%)
    save_folded("conv_y_golden_real_10F.txt", Y_sim_fixed)
    
    save_folded("conv_y_golden_ptb_10F.txt", Y_gold_fixed)
    
    # XUẤT WEIGHT ĐẢO NGƯỢC
    with open("conv_w_input_real_10F.txt", "w") as f:
        for c in range(D_INNER):
            for k in range(KERNEL_SIZE): 
                # Ghi theo thứ tự: W3, W2, W1, W0
                # Hardware đọc tuần tự -> Mem[0]=W3 -> Nhân với x[t] -> Đúng logic!
                w_idx = KERNEL_SIZE - 1 - k
                f.write(to_hex(W_fixed[c, w_idx]) + "\n")
            
    with open("conv_b_input_real_10F.txt", "w") as f:
        for c in range(D_INNER): f.write(to_hex(b_fixed[c]) + "\n")

    print("DONE! Files created for Vivado.")

if __name__ == "__main__":
    run()