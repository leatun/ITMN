import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_MODEL = 64
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# Files Flat (Hex) từ bước trước
FILE_IN   = "phase5_input_flat.txt"
FILE_W    = "phase5_weight_flat.txt"
FILE_GOLD = "phase5_golden_flat.txt"

# --- HELPER ---
def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def hw_mac_vec(vec_in, vec_w):
    """Mô phỏng PE tính MAC"""
    acc = 0
    for i in range(len(vec_in)):
        term = (int(vec_in[i]) * int(vec_w[i])) >> FRAC_BITS
        acc = sat16(acc + term) # Kẹp ngay sau mỗi lần cộng
    return acc

# --- LOAD DATA ---
def load_data():
    print("Loading Data...")
    
    # 1. Load Input (Group-First -> Matrix 128x1000)
    # File Flat: Group -> Token -> Ch16
    # Hardware đọc: Cần ghép lại thành vector 128 kênh cho mỗi Token để tính
    # Tuy nhiên, Hardware đọc theo kiểu Stride Read. Ta cần mô phỏng lại việc lấy input.
    
    with open(FILE_IN) as f: flat_in = [to_signed(l.strip()) for l in f if l.strip()]
    
    # Tái tạo lại cấu trúc RAM A: [Group 0 (1000 lines)]...
    # Mỗi line là vector 16 kênh.
    RAM_A = np.zeros((8, SEQ_LEN, 16), dtype=int)
    idx = 0
    for g in range(8):
        for t in range(SEQ_LEN):
            for k in range(16):
                RAM_A[g, t, k] = flat_in[idx]
                idx += 1
                
    # 2. Load Weights (Chunk-First)
    # File Flat: Chunk -> Col -> Row16
    with open(FILE_W) as f: flat_w = [to_signed(l.strip()) for l in f if l.strip()]
    
    RAM_W = np.zeros((4, D_INNER, 16), dtype=int) # 4 Chunks Output
    idx = 0
    for chunk in range(4):
        for col in range(D_INNER):
            for r in range(16):
                RAM_W[chunk, col, r] = flat_w[idx]
                idx += 1
                
    return RAM_A, RAM_W

# --- SIMULATION ---
def run():
    RAM_A, RAM_W = load_data()
    
    Y_Calc = [] # Sẽ lưu theo Token-First: Token -> Chunk -> Ch16
    
    print("Simulating Phase 5 (Hardware Logic)...")
    
    for t in range(SEQ_LEN): # Loop Token (0..999)
        
        # Bước 1: Gom Input (128 kênh) cho Token t
        # Từ RAM A: Lấy dòng t của Group 0, dòng t của Group 1...
        # -> Vector 128 phần tử
        input_vec = []
        for g in range(8):
            input_vec.extend(RAM_A[g, t, :]) # Nối 16 kênh vào
            
        # Bước 2: Tính toán cho từng Output Chunk
        for chunk in range(4): # Loop Output Group (0..3)
            
            # Tính 16 output kênh trong chunk này
            chunk_out = []
            for r in range(16): # Loop Row trong Chunk
                
                # Lấy vector weight tương ứng với hàng r của chunk này
                # Weight nằm trong RAM_W[chunk, :, r]
                w_vec = RAM_W[chunk, :, r] # Vector 128 phần tử
                
                # Tính MAC
                val = hw_mac_vec(input_vec, w_vec)
                chunk_out.append(val)
                
            Y_Calc.extend(chunk_out)

    # --- COMPARE ---
    print("Comparing with Golden...")
    with open(FILE_GOLD) as f: gold_flat = [to_signed(l.strip()) for l in f if l.strip()]
    
    diffs = []
    for i in range(len(Y_Calc)):
        d = abs(Y_Calc[i] - gold_flat[i])
        diffs.append(d)
        if d > 100 and i < 10:
            print(f"Err Idx {i}: Got {Y_Calc[i]} Exp {gold_flat[i]} Diff {d}")
            
    print(f"Max Diff: {max(diffs)}")
    print(f"Avg Diff: {np.mean(diffs):.2f}")

    if max(diffs) < 100:
        print(">>> SIMULATION PASS! Hardware Logic is CORRECT.")
    else:
        print(">>> SIMULATION FAIL! Check Input/Weight Loading Order.")

if __name__ == "__main__":
    run()