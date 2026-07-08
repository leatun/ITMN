import numpy as np

# --- CẤU HÌNH ---
D_MODEL = 64     # Input Dim
D_INNER = 128    # Output Dim
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE (SỬA LẠI CHO ĐÚNG) ---
BASE_DIR = "D:/DoAn1/Ultility/goldens/"

# Input: File Ma Trận (1000 dòng, 64 cột)
FILE_INPUT  = BASE_DIR + "cpp_golden_files/07_07_MambaBlock_after_norm.txt" 
# Weight: File có index [x,y]
FILE_WEIGHT = BASE_DIR + "golden_vectors_txt/in_proj1_weight.txt" 
# # Bias
# FILE_BIAS   = BASE_DIR + "golden_vectors_txt/in_proj1_bias.txt"
# Golden Output: File Ma Trận (1000 dòng, 128 cột) - Hoặc file 08_X...
FILE_GOLDEN = BASE_DIR + "cpp_golden_files/08_X_after_linear.txt"

# --- HÀM HỖ TRỢ ---
def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

# Hàm parse riêng cho file Weight bị dính index [x,y]
def load_weight_safe(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                try: data.append(float(parts[-1])) # Lấy số cuối cùng
                except: pass
    return np.array(data)

# --- MAIN ---
def run():
    print("=== PREPARE LINEAR REAL DATA (V2 - FIXED) ===")
    
    # 1. LOAD INPUT (Dùng np.loadtxt cho Ma Trận sạch)
    print(f"1. Loading Input from {FILE_INPUT}...")
    try:
        # Load toàn bộ ma trận (1000, 64)
        raw_in = np.loadtxt(FILE_INPUT)
        print(f"   -> Raw Shape: {raw_in.shape}")
        
        # DEBUG: In ra số đầu tiên thực tế
        print(f"   -> DEBUG: First Float in File: {raw_in[0,0]}")
        
        # Lấy Token đầu tiên (Dòng 0, tất cả cột)
        x_vec = raw_in[0, :] 
        
        print(f"   -> Sliced Token 0. Shape: {x_vec.shape}")
        x_fixed = float_to_fixed(x_vec)
        print(f"   -> DEBUG: First Hex: {to_hex(x_fixed[0])}")
        
    except Exception as e:
        print(f"   [ERR] Load Input: {e}"); return

    # 2. LOAD WEIGHT (Dùng hàm parse riêng vì dính index)
    print("2. Loading Weight...")
    try:
        # Load flatten, sau đó reshape (128, 64)
        w_raw = load_weight_safe(FILE_WEIGHT)
        w_mat = w_raw.reshape(D_INNER, D_MODEL)
        w_fixed = float_to_fixed(w_mat)
        print(f"   -> Weight Loaded. Shape: {w_mat.shape}")
    except Exception as e:
        print(f"   [ERR] Load Weight: {e}"); return

    # # 3. LOAD BIAS
    # print("3. Loading Bias...")
    # try:
    #     b_raw = load_weight_safe(FILE_BIAS) # Bias cũng có thể dính index
    #     if len(b_raw) == 0: # Nếu file bias sạch, thử loadtxt
    #          b_raw = np.loadtxt(FILE_BIAS)
    #     b_fixed = float_to_fixed(b_raw)
    #     print(f"   -> Bias Loaded. Shape: {b_raw.shape}")
    # except Exception as e:
    #     print(f"   [ERR] Load Bias: {e}"); return

    # --- RE-ORDERING WEIGHTS (TRANSPOSE CHO PHẦN CỨNG) ---
    print("4. Re-ordering Weights (Folding)...")
    w_hw_ordered = []
    num_chunks = D_INNER // 16 # 8 chunks
    
    for chunk in range(num_chunks):
        start_row = chunk * 16
        end_row = start_row + 16
        # Lấy 16 hàng (16 Output channels)
        w_sub = w_fixed[start_row:end_row, :] # (16, 64)
        # Transpose thành (64, 16) -> (Time, Channel) để ghi vào RAM
        w_sub_T = w_sub.T 
        w_hw_ordered.extend(w_sub_T.flatten())

    # --- XUẤT FILE HEX ---
    print("5. Exporting Files...")
    
    with open("lin_real_x.txt", "w") as f:
        for val in x_fixed: f.write(to_hex(val) + "\n")
        
    with open("lin_real_w.txt", "w") as f:
        for val in w_hw_ordered: f.write(to_hex(val) + "\n")
        
    # with open("lin_real_b.txt", "w") as f:
    #     for val in b_fixed: f.write(to_hex(val) + "\n")

    # 6. LOAD GOLDEN OUTPUT (Token 0)
    # File 08_X có shape (1000, 256) -> Lấy dòng 0, 128 cột đầu
    try:
        gold_raw_all = np.loadtxt(FILE_GOLDEN)
        # Lấy dòng 0, cột 0-127
        y_gold_vec = gold_raw_all[0, :D_INNER]
        y_gold_fixed = float_to_fixed(y_gold_vec)
        
        with open("lin_real_gold.txt", "w") as f:
            for val in y_gold_fixed: f.write(to_hex(val) + "\n")
        print("   -> Golden Exported.")
            
    except Exception as e: print(f"   [WARN] No Golden: {e}")

    print("DONE! Run Vivado simulation now.")

if __name__ == "__main__":
    run()