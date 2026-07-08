import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_MODEL = 64
FRAC_BITS = 12
MAX_INT = 32767
MIN_INT = -32768
SCALE = 1 << 12

# FILES
FILE_DEBUG_IN  = "debug_input_token_first.txt"   # Y_gated
FILE_DEBUG_W   = "debug_weight_reordered.txt"    # W_out
FILE_DEBUG_GOLD = "debug_golden_token_first.txt" # Final Output

# FILE INPUT GỐC (PHASE 1) - Dùng để test Residual
FILE_RAW_INPUT = "linear_x_input.txt" 

def to_signed(val):
    val = int(val)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    return max(min(val, MAX_INT), MIN_INT)

def load_data():
    print("--- Loading Data ---")
    # 1. Y_gated (Input Phase 5)
    with open(FILE_DEBUG_IN) as f: 
        y_gated = np.array([to_signed(int(l.strip(), 16)) for l in f]).reshape(SEQ_LEN, D_INNER)
        
    # 2. Weight (Reconstructed)
    with open(FILE_DEBUG_W) as f: 
        w_flat = [to_signed(int(l.strip(), 16)) for l in f]
    
    W = np.zeros((D_MODEL, D_INNER), dtype=int)
    idx = 0
    for chunk in range(4):
        for col in range(128):
            for r in range(16):
                row = chunk*16 + r
                if idx < len(w_flat): W[row, col] = w_flat[idx]; idx+=1
                
    # 3. Golden Output
    with open(FILE_DEBUG_GOLD) as f:
        gold = np.array([to_signed(int(l.strip(), 16)) for l in f]).reshape(SEQ_LEN, D_MODEL)
        
    # 4. Raw Input (Phase 1) - Cho giả thuyết Residual
    try:
        with open(FILE_RAW_INPUT) as f:
            raw = np.array([to_signed(int(l.strip(), 16)) for l in f]).reshape(SEQ_LEN, 64) # D_MODEL=64
        print("-> Loaded Raw Input (Phase 1) for Residual Check")
    except:
        raw = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
        print("-> WARNING: Could not load Raw Input. Residual Check might fail.")

    return y_gated, W, gold, raw

def run_test():
    Y_gated, W, Gold, Raw_In = load_data()
    
    print("\n--- TEST 1: STANDARD LINEAR (High Precision) ---")
    # Y = Y_gated * W^T
    # Logic: Acc += x*w -> Shift
    Y_Calc = np.zeros_like(Gold)
    for t in range(SEQ_LEN):
        for out_ch in range(D_MODEL):
            acc = 0
            for in_ch in range(D_INNER):
                acc += int(Y_gated[t, in_ch]) * int(W[out_ch, in_ch])
            Y_Calc[t, out_ch] = sat16(acc >> FRAC_BITS)
            
    diff1 = np.abs(Y_Calc - Gold)
    print(f"MAX Diff: {np.max(diff1)}")
    if np.max(diff1) < 100: 
        print("=> MATCH! (Logic Linear thuần túy đúng)")
        return

    print("\n--- TEST 2: LINEAR + RESIDUAL (Cộng Input Phase 1) ---")
    # Y = Linear + Raw_Input
    Y_Res = np.zeros_like(Gold)
    for t in range(SEQ_LEN):
        for ch in range(D_MODEL):
            val = int(Y_Calc[t, ch]) + int(Raw_In[t, ch])
            Y_Res[t, ch] = sat16(val)
            
    diff2 = np.abs(Y_Res - Gold)
    print(f"MAX Diff: {np.max(diff2)}")
    
    if np.max(diff2) < 100:
        print("\n>>> EUREKA! NGUYÊN NHÂN LÀ DO THIẾU RESIDUAL CONNECTION! <<<")
        print("Golden Output bao gồm cả (Linear + Input ban đầu).")
        print("Hardware Phase 5 của cậu mới chỉ tính Linear thôi.")
        print("GIẢI PHÁP: Cậu cần cộng thêm Input (từ RAM A, ADDR_X_INPUT) vào kết quả trước khi ghi ra.")
        return

    print("\n--- TEST 3: TRANSPOSED WEIGHTS (Nghi ngờ file weight bị ngược) ---")
    # Thử nhân với W^T (coi như file text lưu [In, Out])
    # W shape hiện tại (64, 128). Nếu file lưu ngược thì ta cần W.T
    # Nhưng W reconstructed từ file flat đã fix cứng theo logic chunk. 
    # Ta thử giả định W reorder bị sai, quay về W gốc chưa reorder thì phức tạp.
    # Bỏ qua Test 3 nếu Test 2 fail, vì khả năng cao là Test 2.
    
    print("\n--- ANALYSIS ---")
    print(f"Tok0 Ch0 | Calc Linear: {Y_Calc[0,0]}")
    print(f"Tok0 Ch0 | Raw Input:   {Raw_In[0,0]}")
    print(f"Tok0 Ch0 | Sum:         {sat16(Y_Calc[0,0] + Raw_In[0,0])}")
    print(f"Tok0 Ch0 | Golden:      {Gold[0,0]}")

if __name__ == "__main__":
    run_test()