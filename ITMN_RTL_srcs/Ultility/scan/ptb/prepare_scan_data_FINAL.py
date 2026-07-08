import numpy as np
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

# --- ĐƯỜNG DẪN FILE GỐC C++ (SỬA LẠI CHO ĐÚNG) ---
# 1. Delta (Sau Softplus): File 10_09...
F_DELTA_RAW = "D:/DoAn1/Ultility/goldens/cpp_golden_files/10_09_Mixer_delta_final.txt"

# 2. X Input (Sau Conv+SiLU): File 09_08...
F_X_RAW     = "D:/DoAn1/Ultility/goldens/cpp_golden_files/09_08_Mixer_x_activated.txt"

# 3. Gate Raw (Trước SiLU): File linear2_golden (như cậu nói)
# Hãy chắc chắn đây là Z chưa qua SiLU. Nếu file này là output của in_proj2 thì đúng.
F_GATE_RAW  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/08_X_after_linear.txt" 
# Lưu ý: Nếu file 08_X chứa cả X và Z (256 cột), script sẽ tự cắt.

# 4. A Parameter (A_log): File A_log
F_A_LOG     = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/A_log.txt"

# 5. D Parameter: File D
F_D_RAW     = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/D.txt"

# 6. B & C Raw (Shared): File 11_10 và 12_11
F_B_RAW     = "D:/DoAn1/Ultility/goldens/cpp_golden_files/11_10_Mixer_B_raw.txt"
F_C_RAW     = "D:/DoAn1/Ultility/goldens/cpp_golden_files/12_11_Mixer_C_raw.txt"

# 7. Golden Output (Y Gated): File 14_Mixer_y_gated
F_Y_GOLDEN  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"


# --- HELPER ---
def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def load_and_transpose(filepath, name, shape_hint, slice_col=None):
    print(f"Processing {name}...")
    try:
        raw = np.loadtxt(filepath)
        
        # Xử lý cắt cột (cho Gate/Z nếu file chứa cả X và Z)
        if slice_col is not None:
            # slice_col = (start, end)
            if raw.ndim == 2 and raw.shape[1] >= slice_col[1]:
                print(f"   -> Slicing columns {slice_col[0]}:{slice_col[1]}")
                raw = raw[:, slice_col[0]:slice_col[1]]
        
        # Xử lý Transpose:
        # Nếu shape là (1000, 128) -> Transpose thành (128, 1000)
        # Mục tiêu: [Channel][Time]
        if raw.shape == (SEQ_LEN, D_INNER):
            print(f"   -> Transposing from {raw.shape} to ({D_INNER}, {SEQ_LEN})")
            data = raw.T
        elif raw.shape == (D_INNER, SEQ_LEN):
            data = raw
        elif raw.shape == (D_INNER,): # Vector D
            data = raw
        elif raw.shape == (D_INNER * D_STATE,): # A_log flatten
            data = raw.reshape(D_INNER, D_STATE)
        elif raw.shape == (SEQ_LEN, D_STATE): # B, C shared
            data = raw
        else:
            print(f"   [WARN] Unexpected shape: {raw.shape}")
            data = raw
            
        return data
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None

def run():
    print("=== DATA PREPARATION FOR SCAN CORE (AUTO TRANSPOSE) ===")
    
    # 1. DELTA
    delta = load_and_transpose(F_DELTA_RAW, "Delta", (SEQ_LEN, D_INNER))
    if delta is not None:
        with open("scan_real_delta.txt", "w") as f:
            for d in range(SEQ_LEN):
                for t in range(D_INNER): f.write(to_hex(float_to_fixed(delta[d, t])) + "\n")

    # 2. X INPUT
    x_in = load_txt_special(F_X_RAW, "X Input", is_gate=False)
    if x_in is not None:
        with open("scan_real_x.txt", "w") as f:
            for d in range(D_INNER):
                for t in range(SEQ_LEN): f.write(to_hex(float_to_fixed(x_in[d, t])) + "\n")

    # 3. GATE INPUT (Z RAW)
    # File 08_X có 256 cột. Z là cột 128-255.
    gate = load_txt_special(F_GATE_RAW, "Gate Raw (Z)", is_gate=True)
    if gate is not None:
        with open("scan_real_gate.txt", "w") as f:
            for d in range(D_INNER):
                for t in range(SEQ_LEN): f.write(to_hex(float_to_fixed(gate[d, t])) + "\n")

    # 4. A PARAMETER
    # Parse file A_log (bỏ index [x,y])
    a_vals = []
    with open(F_A_LOG, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts: a_vals.append(float(parts[-1]))
    A_real = -np.exp(np.array(a_vals).reshape(D_INNER, D_STATE))
    with open("scan_real_A.txt", "w") as f:
        for d in range(D_INNER):
            for n in range(D_STATE): f.write(to_hex(float_to_fixed(A_real[d, n])) + "\n")
    print("Processed A Parameter.")

    # 5. D PARAMETER
    # Parse file D (bỏ index [x])
    d_vals = []
    with open(F_D_RAW, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts: d_vals.append(float(parts[-1]))
    D_real = np.array(d_vals)
    with open("scan_real_D.txt", "w") as f:
        for val in D_real: f.write(to_hex(float_to_fixed(val)) + "\n")
    print("Processed D Parameter.")

    # 6. B & C RAW
    # Không cần transpose channel, chỉ load (1000, 16)
    b_raw = np.loadtxt(F_B_RAW)
    with open("scan_real_B_shared.txt", "w") as f:
        for t in range(SEQ_LEN):
            for n in range(D_STATE): f.write(to_hex(float_to_fixed(b_raw[t, n])) + "\n")
    
    c_raw = np.loadtxt(F_C_RAW)
    with open("scan_real_C_shared.txt", "w") as f:
        for t in range(SEQ_LEN):
            for n in range(D_STATE): f.write(to_hex(float_to_fixed(c_raw[t, n])) + "\n")
    print("Processed B & C Shared.")

    # 7. GOLDEN Y
    y_gold = load_and_transpose(F_Y_GOLDEN, "Golden Y", (SEQ_LEN, D_INNER))
    if y_gold is not None:
        with open("scan_real_y_golden.txt", "w") as f:
            for d in range(D_INNER):
                for t in range(SEQ_LEN): f.write(to_hex(float_to_fixed(y_gold[d, t])) + "\n")

    print("\nDONE! All files generated. Run verify_scan_core_math.py now.")

def load_txt_special(path, name, is_gate=False):
    print(f"Processing {name}...")
    try:
        raw = np.loadtxt(path)
        # Nếu file là 08_X (1000, 256)
        if raw.shape[1] == 256:
            if is_gate: # Lấy Z (128 sau)
                print("   -> Slicing Z (Cols 128-255)")
                data = raw[:, 128:]
            else: # Lấy X (128 đầu)
                print("   -> Slicing X (Cols 0-127)")
                data = raw[:, :128]
        else:
            data = raw
            
        # Transpose (1000, 128) -> (128, 1000)
        return data.T
    except Exception as e:
        print(f"   [ERROR] {e}")
        return None

if __name__ == "__main__":
    run()