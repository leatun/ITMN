import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128  # Input Dimension
D_MODEL = 64   # Output Dimension
FRAC_BITS = 12
MAX_INT = 32767
MIN_INT = -32768

# --- FILES (FLAT HEX) ---
FILE_IN_FLAT   = "phase5_input_flat.txt"
FILE_W_FLAT    = "phase5_weight_flat.txt"
FILE_GOLD_FLAT = "phase5_golden_flat.txt"

# --- HELPER FUNCTIONS ---
def to_signed(hex_str):
    val = int(hex_str.strip(), 16)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

# --- HARDWARE MAC SIMULATION ---
def hw_mac_dot_product(vec_x, vec_w):
    """
    Mô phỏng phép nhân chập (Dot Product) của Hardware.
    Input: 2 vector int16.
    Logic: (x * w) >> 12, sau đó cộng dồn.
    """
    acc = 0
    for i in range(len(vec_x)):
        # 1. Nhân 2 số 16-bit -> ra 32-bit
        prod = int(vec_x[i]) * int(vec_w[i])
        
        # 2. Dịch bit (Shift Right 12) để về lại dạng Q3.12
        # Cậu lưu ý: Có 2 kiểu PE:
        # Kiểu A: Acc += (Prod >> 12)  <-- (Thường dùng cho tiết kiệm bit Accumulator)
        # Kiểu B: Acc += Prod; Sau đó Output = Acc >> 12 <-- (Chính xác hơn)
        
        # Dựa trên câu "dịch bit xong mới nhân" (khả năng ý cậu là nhân xong mới dịch)
        # Tôi sẽ dùng Kiểu A vì nó phổ biến trong các module PE đơn giản.
        term = prod >> FRAC_BITS
        
        # 3. Cộng dồn (Accumulate)
        acc += term
        
        # *Lưu ý*: Nếu PE của cậu có Saturation ngay sau mỗi lần cộng, 
        # hãy uncomment dòng dưới:
        # acc = sat16(acc) 
        
    # 4. Kẹp đầu ra cuối cùng về 16-bit
    return sat16(acc)

def load_and_run():
    print("--- 1. Reconstructing Matrices from Flat Files ---")
    
    # 1. LOAD INPUT X (1000, 128)
    # Logic file flat: Group(8) -> Token(1000) -> Ch(16)
    X = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    try:
        with open(FILE_IN_FLAT) as f:
            lines = f.readlines()
        
        idx = 0
        for g in range(8):
            for t in range(SEQ_LEN):
                for k in range(16):
                    col = g * 16 + k
                    X[t, col] = to_signed(lines[idx])
                    idx += 1
        print(f"   -> Input X Loaded: Shape {X.shape}")
    except Exception as e: print(f"Err loading X: {e}")

    # 2. LOAD WEIGHT W (64, 128)
    # Logic file flat: Chunk(4) -> Col(128) -> Row(16)
    W = np.zeros((D_MODEL, D_INNER), dtype=int)
    try:
        with open(FILE_W_FLAT) as f:
            lines = f.readlines()
        
        idx = 0
        num_chunks = D_MODEL // 16 # 4
        for chunk in range(num_chunks):
            for col in range(D_INNER):
                for r in range(16):
                    row = chunk * 16 + r
                    W[row, col] = to_signed(lines[idx])
                    idx += 1
        print(f"   -> Weight W Loaded: Shape {W.shape}")
    except Exception as e: print(f"Err loading W: {e}")

    # 3. LOAD GOLDEN (1000, 64)
    # Logic file flat: Token(1000) -> Chunk(4) -> Ch(16)
    Y_Gold = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
    try:
        with open(FILE_GOLD_FLAT) as f:
            lines = f.readlines()
        idx = 0
        for t in range(SEQ_LEN):
            for chunk in range(4):
                for k in range(16):
                    col = chunk * 16 + k
                    Y_Gold[t, col] = to_signed(lines[idx])
                    idx += 1
        print(f"   -> Golden Loaded: Shape {Y_Gold.shape}")
    except Exception as e: print(f"Err loading Golden: {e}")

    # --- 4. RUN SIMULATION ---
    print("\n--- 2. Running Software Linear (Fixed-Point) ---")
    Y_Calc = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
    
    for t in range(SEQ_LEN):
        for out_ch in range(D_MODEL):
            # Tính dot product hàng 't' của X với cột 'out_ch' của W
            # Lưu ý: W trong công thức y = xW^T là (64, 128)
            # Nên ta nhân x[t] (128) với W[out_ch] (128)
            val = hw_mac_dot_product(X[t], W[out_ch])
            Y_Calc[t, out_ch] = val
            
        if t % 200 == 0: print(f"   Processed Token {t}...")

    # --- 5. COMPARE ---
    print("\n--- 3. Comparison ---")
    diff = np.abs(Y_Calc - Y_Gold)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"MAX Diff: {max_diff}")
    print(f"AVG Diff: {avg_diff:.2f}")
    
    if max_diff == 0:
        print("\n>>> KẾT QUẢ: TUYỆT ĐỐI KHỚP! Logic Phần mềm OK. <<<")
    elif max_diff <= 2:
        print("\n>>> KẾT QUẢ: KHỚP (Sai số làm tròn 1-2 bit). Chấp nhận được. <<<")
    else:
        print("\n>>> KẾT QUẢ: KHÔNG KHỚP! <<<")
        print("Có thể do cách làm tròn (Round vs Floor) hoặc thứ tự cộng.")
        # Debug 1 dòng lỗi
        for t in range(SEQ_LEN):
            for ch in range(D_MODEL):
                if diff[t, ch] > 2:
                    print(f"First Error at Token {t}, Ch {ch}:")
                    print(f"   Calc: {Y_Calc[t, ch]}")
                    print(f"   Gold: {Y_Gold[t, ch]}")
                    print(f"   Diff: {diff[t, ch]}")
                    return

if __name__ == "__main__":
    load_and_run()