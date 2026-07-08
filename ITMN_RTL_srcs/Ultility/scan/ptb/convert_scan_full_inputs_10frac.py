import numpy as np
import math
import os

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
DATA_WIDTH = 16
FRAC_BITS = 10
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FOLDER CHỨA FILE C++ ---
CPP_DIR = "D:/DoAn1/Ultility/goldens"

# Tên file gốc (Cậu check lại xem đúng tên chưa nha)
F_A_LOG       = "golden_vectors_txt/A_log.txt"       # Shape (128, 16)
F_D           = "golden_vectors_txt/D.txt"           # Shape (128,)
F_DELTA       = "cpp_golden_files/10_09_Mixer_delta_final.txt"        # Shape (128, 1000)
F_X_CONV_SILU = "cpp_golden_files/09_08_Mixer_x_activated.txt"        # Shape (128, 1000) - Input X cho Scan
F_B_RAW       = "cpp_golden_files/11_10_Mixer_B_raw.txt"              # Shape (1000, 16)
F_C_RAW       = "cpp_golden_files/12_11_Mixer_C_raw.txt"              # Shape (1000, 16)
# F_Y_GOLDEN    = "cpp_golden_files/15_Mixer_final_output.txt"          # Shape (128, 1000) - Output cuối cùng (sau Gating)

# --- HELPER ---
def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def load_txt(filename, expected_shape, name="Data"):
    path = os.path.join(CPP_DIR, filename)
    print(f"Loading {name} from {path}...")
    try:
        raw = np.loadtxt(path)
        # Xử lý trường hợp file lưu dạng flatten hoặc transpose
        if raw.size != np.prod(expected_shape):
            print(f"  [WARN] Size mismatch! File: {raw.size}, Exp: {np.prod(expected_shape)}")
        
        # Reshape linh hoạt (thử các chiều)
        if len(expected_shape) == 2:
            if raw.shape == expected_shape:
                data = raw
            elif raw.shape == (expected_shape[1], expected_shape[0]):
                data = raw.T # Transpose nếu bị ngược
            else:
                data = raw.reshape(expected_shape)
        else:
            data = raw.reshape(expected_shape)
            
        print(f"  -> Shape OK: {data.shape}")
        return data
    except Exception as e:
        print(f"  [ERROR] Load failed: {e}")
        return None

# --- MAIN CONVERTER ---
def run():
    print("=== CONVERTING SCAN CORE DATA (FULL 128 CHANNELS) ===")
    
    # 1. A_log -> A
    # Hardware công thức: exp(delta * A)
    # Trong Mamba gốc: parameter là A_log. A = -exp(A_log)
    a_log = load_txt(F_A_LOG, (D_INNER, D_STATE), "A_log")
    if a_log is not None:
        A_real = -np.exp(a_log) # Tính A thật
        A_fixed = float_to_fixed(A_real)
        
        with open("scan_real_A_10F.txt", "w") as f:
            for d in range(D_INNER):
                for n in range(D_STATE):
                    f.write(to_hex(A_fixed[d, n]) + "\n")
        print("  -> Exported scan_real_A_10F.txt (Flatten D_INNER x D_STATE)")

    # 2. D
    d_val = load_txt(F_D, (D_INNER,), "D")
    if d_val is not None:
        D_fixed = float_to_fixed(d_val)
        with open("scan_real_D_10F.txt", "w") as f:
            for val in D_fixed:
                f.write(to_hex(val) + "\n")
        print("  -> Exported scan_real_D_10F.txt")

    # 3. Delta
    delta = load_txt(F_DELTA, (D_INNER, SEQ_LEN), "Delta")
    if delta is not None:
        Delta_fixed = float_to_fixed(delta)
        # Xuất theo thứ tự: Channel 0 (T0..999), Channel 1...
        with open("scan_real_delta_10F.txt", "w") as f:
            for t in range(SEQ_LEN):
                for d in range(D_INNER):
                    f.write(to_hex(Delta_fixed[d, t]) + "\n")
        print("  -> Exported scan_real_delta_10F.txt (Flatten Channel x Time)")

    # 4. X Input (X Activated)
    x_in = load_txt(F_X_CONV_SILU, (D_INNER, SEQ_LEN), "X Input (Conv+SiLU)")
    if x_in is not None:
        X_fixed = float_to_fixed(x_in)
        with open("scan_real_x_10F.txt", "w") as f:
            for d in range(D_INNER):
                for t in range(SEQ_LEN):
                    f.write(to_hex(X_fixed[d, t]) + "\n")
        print("  -> Exported scan_real_x_10F.txt")

    # 5. B_raw (Shared)
    # Shape: (1000, 16)
    b_raw = load_txt(F_B_RAW, (SEQ_LEN, D_STATE), "B Raw")
    if b_raw is not None:
        B_fixed = float_to_fixed(b_raw)
        with open("scan_real_B_shared_10F.txt", "w") as f:
            for t in range(SEQ_LEN):
                for n in range(D_STATE):
                    f.write(to_hex(B_fixed[t, n]) + "\n")
        print("  -> Exported scan_real_B_shared_10F.txt")

    # 6. C_raw (Shared)
    c_raw = load_txt(F_C_RAW, (SEQ_LEN, D_STATE), "C Raw")
    if c_raw is not None:
        C_fixed = float_to_fixed(c_raw)
        with open("scan_real_C_shared_10F.txt", "w") as f:
            for t in range(SEQ_LEN):
                for n in range(D_STATE):
                    f.write(to_hex(C_fixed[t, n]) + "\n")
        print("  -> Exported scan_real_C_shared_10F.txt")

    # 7. Golden Output (Mixer Final)
    # Lưu ý: File 15_Mixer_final_output là kết quả sau khi Gating và OutProj?
    # Trong scan_flow.txt cậu gửi, y_gated là output của loop scan.
    # Còn Mixer_final_output thường là sau lớp Linear OutProj.
    # CẬU CẦN CHECK LẠI: Cậu muốn test Output ngay sau Gating (y_gated) hay sau cùng?
    # Nếu test Scan Core, ta cần so sánh với y_gated (File 14_Mixer_y_gated).
    
    # Giả sử lấy file 14_Mixer_y_gated (nếu có, hoặc dùng tạm file 15 nhưng coi chừng sai)
    # Tốt nhất là dùng file "14_Mixer_y_gated.txt" nếu cậu đã trích xuất.
    # Nếu chưa có file 14, hãy thêm vào script extract.
    
    # Ở đây tôi dùng placeholder file 14
    F_Y_GATED = "cpp_golden_files/1014_14_Mixer_y_gated.txt" 
    path_y = os.path.join(CPP_DIR, F_Y_GATED)
    
    if os.path.exists(path_y):
        y_gated = load_txt(F_Y_GATED, (D_INNER, SEQ_LEN), "Y Gated (Golden)")
        if y_gated is not None:
            Y_fixed = float_to_fixed(y_gated)
            with open("scan_real_y_golden_10F.txt", "w") as f:
                for d in range(D_INNER):
                    for t in range(SEQ_LEN):
                        f.write(to_hex(Y_fixed[d, t]) + "\n")
            print("  -> Exported scan_real_y_golden_10F.txt")
    else:
        print(f"  [WARN] File {F_Y_GATED} not found! Cannot generate golden output.")

if __name__ == "__main__":
    run()