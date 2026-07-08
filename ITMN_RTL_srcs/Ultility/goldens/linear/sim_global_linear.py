import numpy as np

# --- CẤU HÌNH ---
D_MODEL = 64     # Input Dim
D_INNER = 128    # Output Dim
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE GỐC (C++ Raw Float) ---
BASE_DIR = "D:/DoAn1/Ultility/goldens/"
# Input: Lấy từ file 07_07 (Sau Norm) hoặc 06_06 tùy cậu chọn, ở đây dùng 07_07 cho đúng luồng
FILE_INPUT  = BASE_DIR + "cpp_golden_files/07_07_MambaBlock_after_norm.txt"
# Weight: File weight gốc (có thể dính index [x,y])
FILE_WEIGHT = BASE_DIR + "golden_vectors_txt/in_proj2_weight.txt"

# Output files cho Verilog
OUT_X = "sys_lin_x.txt"
OUT_W = "sys_lin_w_2.txt"
OUT_GOLD = "sys_lin_gold_2.txt"

# --- HÀM HỖ TRỢ ---
def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

# Hàm load weight an toàn
def load_weight_safe(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                try: data.append(float(parts[-1]))
                except: pass
    return np.array(data).reshape(D_INNER, D_MODEL)

def run():
    print("=== SIMULATING GLOBAL CONTROLLER LINEAR ===")
    
    # 1. LOAD INPUT (Token 0)
    try:
        raw_in = np.loadtxt(FILE_INPUT)
        # Lấy dòng đầu tiên (Token 0)
        x_vec_float = raw_in[0, :] 
        x_fixed = float_to_fixed(x_vec_float)
        print(f"Input Loaded. Shape: {x_vec_float.shape}")
    except Exception as e: print(f"Err Input: {e}"); return

    # 2. LOAD WEIGHT
    try:
        w_mat_float = load_weight_safe(FILE_WEIGHT)
        w_fixed = float_to_fixed(w_mat_float)
        print(f"Weight Loaded. Shape: {w_mat_float.shape} (128, 64)")
    except Exception as e: print(f"Err Weight: {e}"); return

    # 3. MÔ PHỎNG VÀ XUẤT FILE
    # Global Controller chạy theo thứ tự:
    # Loop Chunk 0..7:
    #    Loop Feed 0..63:
    #       Tính toán...
    
    hw_w_list = []      # List chứa weight theo thứ tự controller đọc
    hw_gold_list = []   # List chứa output
    
    print("Simulating Hardware Logic...")
    
    # --- LOOP CHUNKS (8 Chunks) ---
    for chunk in range(8):
        start_ch = chunk * 16
        end_ch   = start_ch + 16
        
        # Lấy 16 hàng weight tương ứng với 16 kênh output này
        # Shape: (16, 64)
        w_chunk = w_fixed[start_ch:end_ch, :]
        
        # Mô phỏng PE Accumulators cho 16 kênh
        acc = np.zeros(16, dtype=int)
        
        # --- LOOP FEED (64 Inputs) ---
        # Controller đọc lần lượt từng cột của Weight matrix
        for k in range(D_MODEL):
            x_val = x_fixed[k]
            
            # Lấy cột k của w_chunk (16 giá trị)
            # Đây chính là giá trị w_in_vec mà Linear Layer nhận ở nhịp k
            w_col = w_chunk[:, k]
            
            # Lưu vào list để xuất file Hex (Controller đọc tuần tự cái này)
            # Mỗi dòng file Hex sẽ chứa 16 số này (nhưng ta lưu flatten, TB sẽ pack)
            for w in w_col:
                hw_w_list.append(w)
            
            # Tính toán MAC (Mô phỏng PE)
            # acc += (x * w) >> 12
            for i in range(16):
                prod = (int(x_val) * int(w_col[i])) >> FRAC_BITS
                acc[i] = sat16(acc[i] + prod)
                
        # Sau khi xong 64 nhịp, ta có 16 output hoàn chỉnh
        # Thêm vào danh sách Golden
        for val in acc:
            hw_gold_list.append(val)

    # 4. GHI FILE
    print("Exporting Files...")
    
    # File X: 64 dòng
    with open(OUT_X, "w") as f:
        for val in x_fixed: f.write(to_hex(val) + "\n")
        
    # File W: 8 chunks * 64 feeds * 16 values = 8192 dòng
    with open(OUT_W, "w") as f:
        for val in hw_w_list: f.write(to_hex(val) + "\n")
        
    # File Gold: 8 chunks * 16 outputs = 128 dòng
    with open(OUT_GOLD, "w") as f:
        for val in hw_gold_list: f.write(to_hex(val) + "\n")

    print(f"DONE! Files created:")
    print(f"  {OUT_X} (Input)")
    print(f"  {OUT_W} (Ordered Weights for Controller)")
    print(f"  {OUT_GOLD} (Hardware Golden Output - Bias=0)")

if __name__ == "__main__":
    run()