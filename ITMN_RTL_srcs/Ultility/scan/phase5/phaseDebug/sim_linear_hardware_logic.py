import numpy as np

# ==============================================================================
# 1. CẤU HÌNH & HẰNG SỐ
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128  # Input
D_MODEL = 64   # Output
FRAC_BITS = 12
MAX_INT = 32767
MIN_INT = -32768

# Files Input (Dùng bộ file Debug Token-First cậu vừa tạo)
FILE_IN_HEX     = "final/debug_input_correct.txt"
FILE_W_HEX      = "final/debug_weight_correct.txt"
FILE_GOLDEN_HEX = "final/debug_golden_correct.txt"

# ==============================================================================
# 2. HÀM MÔ PHỎNG PHẦN CỨNG (BIT-EXACT)
# ==============================================================================

def to_signed(hex_str):
    """Chuyển Hex string sang Signed Integer 16-bit"""
    val = int(hex_str.strip(), 16)
    return val - 65536 if val & 0x8000 else val

def sat16(val):
    """Mô phỏng bộ bão hòa (Saturation) đầu ra"""
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def hw_mac_step(acc, x, w):
    """
    Mô phỏng 1 bước MAC (Multiply-Accumulate) trong PE.
    Logic: Acc_new = Acc_old + ( (x * w) >> 12 )
    """
    # 1. Nhân 2 số 16-bit -> 32-bit
    prod = int(x) * int(w)
    
    # 2. Dịch bit (Arithmetic Shift Right)
    term = prod >> FRAC_BITS
    
    # 3. Cộng dồn
    # Lưu ý: Trong Linear, thường ta cộng dồn trong thanh ghi rộng (vd 32-bit)
    # rồi mới kẹp ở cuối cùng. Nếu PE của cậu kẹp sau mỗi lần cộng, 
    # hãy uncomment dòng 'return sat16(new_acc)'
    new_acc = acc + term
    
    # return sat16(new_acc) # <-- Nếu PE kẹp liên tục
    return new_acc          # <-- Nếu PE dùng Accumulator 32-bit (thường dùng cho Linear)

# Thay thế hàm cũ bằng hàm này trong script Python
def hw_mac_dot_product(vec_x, vec_w):
    """
    Mô phỏng High Precision Accumulation (Logic của C++/Pytorch)
    """
    acc_huge = 0 # Thanh ghi lớn (32-bit hoặc hơn)
    
    for i in range(len(vec_x)):
        # 1. Nhân 2 số 16-bit -> 32-bit
        prod = int(vec_x[i]) * int(vec_w[i])
        
        # 2. CỘNG DỒN LUÔN (Chưa dịch bit vội!)
        # Đây là sự khác biệt: Giữ nguyên độ chính xác
        acc_huge += prod 
        
    # 3. Dịch bit MỘT LẦN DUY NHẤT ở cuối cùng
    final_res = acc_huge >> FRAC_BITS
    
    # 4. Kẹp
    return sat16(final_res)

# ==============================================================================
# 3. LOAD DỮ LIỆU TỪ FILE HEX
# ==============================================================================

def load_data():
    print("--- Loading Data ---")
    
    # 1. Load Input (Token-First)
    # File phẳng: Token 0 (128 dòng), Token 1...
    with open(FILE_IN_HEX, 'r') as f:
        x_flat = [to_signed(line) for line in f if line.strip()]
    X = np.array(x_flat).reshape(SEQ_LEN, D_INNER)
    print(f"Input Loaded: {X.shape}")

    # 2. Load Weight (Reordered Chunk-First)
    # File phẳng: Chunk 0 (Col 0..127) -> Chunk 1...
    # Trong mỗi Col có 16 Rows.
    # Cần tái tạo lại ma trận W thực tế để tính toán
    with open(FILE_W_HEX, 'r') as f:
        w_flat = [to_signed(line) for line in f if line.strip()]
    
    # Tái tạo lại ma trận W (64, 128) từ file Reordered
    # File cấu trúc: Chunk(4) -> Col(128) -> Row(16)
    W_Reconstructed = np.zeros((D_MODEL, D_INNER), dtype=int)
    
    idx = 0
    num_chunks = D_MODEL // 16 # 4 chunks
    for chunk in range(num_chunks):
        start_row = chunk * 16
        for col in range(D_INNER): # 128 cols
            for r in range(16):    # 16 rows trong 1 chunk
                row = start_row + r
                if idx < len(w_flat):
                    W_Reconstructed[row, col] = w_flat[idx]
                    idx += 1
    print(f"Weight Loaded & Reconstructed: {W_Reconstructed.shape}")

    # 3. Load Golden
    with open(FILE_GOLDEN_HEX, 'r') as f:
        gold_flat = [to_signed(line) for line in f if line.strip()]
    Y_Gold = np.array(gold_flat).reshape(SEQ_LEN, D_MODEL)
    print(f"Golden Loaded: {Y_Gold.shape}")
    
    return X, W_Reconstructed, Y_Gold

# ==============================================================================
# 4. CHẠY MÔ PHỎNG VÀ SO SÁNH
# ==============================================================================

def run_simulation_hw():
    X, W, Y_Gold = load_data()
    
    print("\n--- Running Bit-Exact Simulation ---")
    Y_Sim = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
    
    # Loop Token
    for t in range(SEQ_LEN):
        # Loop Output Channel (0..63)
        for out_ch in range(D_MODEL):
            
            # Tính Dot Product: Row X[t] * Col W[out_ch]
            acc = 0
            
            # Giả lập Bias = 0 (như code C++)
            
            for in_ch in range(D_INNER):
                x_val = X[t, in_ch]
                w_val = W[out_ch, in_ch] # Lưu ý W shape (Out, In)
                
                # Thực hiện phép tính y hệt PE
                acc = hw_mac_step(acc, x_val, w_val)
            
            # Kẹp đầu ra cuối cùng
            Y_Sim[t, out_ch] = sat16(acc)
            
        if t % 200 == 0: print(f"Processed Token {t}...")

    print("\n--- Comparison Results ---")
    
    # Tính sai số
    diff = np.abs(Y_Sim - Y_Gold)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"MAX Diff: {max_diff}")
    print(f"AVG Diff: {avg_diff:.2f}")
    
    print("\n--- Detail Check (Token 0) ---")
    print(f"Sim HW (Tok0, Ch0..9): {Y_Sim[0, :10]}")
    print(f"Golden (Tok0, Ch0..9): {Y_Gold[0, :10]}")
    print(f"Diff   (Tok0, Ch0..9): {diff[0, :10]}")

    if max_diff == 0:
        print("\n>>> RESULT: MATCH PERFECTLY! Hardware Logic is 100% Correct. <<<")
    elif max_diff <= 64:
        print("\n>>> RESULT: MATCH WITH PRECISION ERROR. Hardware Logic is Likely Correct. <<<")
        print("Lý do: Cộng dồn 128 lần, mỗi lần mất phần thập phân do dịch bit.")
        print("Sai số tối đa chấp nhận được ~ 0.5 * 128 = 64 đơn vị.")
    else:
        print("\n>>> RESULT: MISMATCH! Something is wrong with Logic or Weight Order. <<<")



# ==============================================================================
# 4. CHẠY MÔ PHỎNG (LOGIC "NHÀ GIÀU" - HIGH PRECISION)
# ==============================================================================

def run_simulation_py():
    X, W, Y_Gold = load_data()
    
    print("\n--- Running High-Precision Simulation (Add then Shift) ---")
    Y_Sim = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
    
    # Loop Token
    for t in range(SEQ_LEN):
        # Loop Output Channel (0..63)
        for out_ch in range(D_MODEL):
            
            # --- KHỞI TẠO ACCUMULATOR ---
            # Dùng biến này để chứa tổng cực lớn (chưa dịch bit)
            acc_huge = 0 
            
            # Loop Input Channel (0..127)
            for in_ch in range(D_INNER):
                x_val = int(X[t, in_ch])
                w_val = int(W[out_ch, in_ch]) 
                
                # --- BƯỚC 1: NHÂN VÀ CỘNG DỒN LUÔN (KHÔNG DỊCH BIT) ---
                # Logic C++/Pytorch chuẩn: acc += x * w
                acc_huge += (x_val * w_val)
            
            # --- BƯỚC 2: DỊCH BIT MỘT LẦN DUY NHẤT Ở CUỐI ---
            # Bây giờ mới đưa về Q3.12
            final_res = acc_huge >> FRAC_BITS
            
            # --- BƯỚC 3: KẸP ---
            Y_Sim[t, out_ch] = sat16(final_res)
            
        if t % 200 == 0: print(f"Processed Token {t}...")

    print("\n--- Comparison Results (High Precision Logic) ---")
    
    # Tính sai số
    diff = np.abs(Y_Sim - Y_Gold)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"MAX Diff: {max_diff}")
    print(f"AVG Diff: {avg_diff:.2f}")
    
    print("\n--- Detail Check (Token 0) ---")
    print(f"Sim HW (Tok0, Ch0..9): {Y_Sim[0, :10]}")
    print(f"Golden (Tok0, Ch0..9): {Y_Gold[0, :10]}")
    print(f"Diff   (Tok0, Ch0..9): {diff[0, :10]}")

    if max_diff <= 2:
        print("\n>>> RESULT: MATCH! <<<")
        print("Nguyên nhân sai lệch đã rõ: Hardware dùng Low Precision (Shift-then-Add),")
        print("còn Golden dùng High Precision (Add-then-Shift).")
        print("Hardware của cậu KHÔNG SAI về logic, chỉ khác về kiến trúc Accumulator.")
    else:
        print("\n>>> RESULT: STILL MISMATCH! <<<")
        print("Có vẻ vấn đề không chỉ nằm ở độ chính xác. Cần kiểm tra lại thứ tự Weight/Input.")
        
        
def run_pure_math_simulation():
    X, W, Y_Gold = load_data()
    
    print("\n--- Running Pure Math Simulation (Numpy Matrix Multiplication) ---")
    
    # Chuyển sang int64 để nhân ma trận không bị tràn số, đảm bảo độ chính xác tuyệt đối
    X_64 = X.astype(np.int64)
    W_64 = W.astype(np.int64)
    
    # 1. Tính toán Toán học thuần túy: Y = X * W^T
    # Kết quả lúc này đang ở dạng Q6.24 (do Q3.12 nhân Q3.12)
    # Tương đương với việc cộng dồn trong thanh ghi 64-bit mà không mất mát gì.
    Y_Huge = np.matmul(X_64, W_64.T)
    
    # 2. Dịch bit về lại Q3.12 (Chia cho 4096)
    Y_Calc = Y_Huge >> FRAC_BITS
    
    # 3. Kẹp biên 16-bit (Để khớp với format file Golden)
    Y_Calc = np.clip(Y_Calc, MIN_INT, MAX_INT)

    # --- SO SÁNH ---
    print("\n--- Comparison Results ---")
    
    diff = np.abs(Y_Calc - Y_Gold)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"MAX Diff: {max_diff}")
    print(f"AVG Diff: {avg_diff:.2f}")
    
    print("\n--- Detail Check (Token 0, Channel 0) ---")
    print(f"Calc (Pure Math): {Y_Calc[0, 0]}")
    print(f"Golden (File)   : {Y_Gold[0, 0]}")
    print(f"Diff            : {diff[0, 0]}")

    # --- KIỂM TRA BIAS (QUAN TRỌNG) ---
    # Nếu Diff lớn, ta tính xem trung bình độ lệch là bao nhiêu.
    # Nếu độ lệch này khác 0 và ổn định, đó chính là BIAS bị thiếu.
    bias_estimate = np.mean(Y_Gold - Y_Calc, axis=0)
    print("\n--- Estimated Bias (First 5 channels) ---")
    print(f"Bias Hex: {[int(b) for b in bias_estimate[:5]]}")
    print("Nếu các số này khác 0, file Golden có cộng thêm Bias (hoặc Residual) mà Input không có.")

    if max_diff <= 2:
        print("\n>>> RESULT: MATCH! (File Hex chuẩn, Logic chuẩn) <<<")
    else:
        print("\n>>> RESULT: MISMATCH! <<<")
        print("Dữ liệu trong các file Hex không khớp với công thức Linear thuần túy (Y=XW).")
        print("Khả năng cao Golden Output chứa Bias hoặc Residual.")

if __name__ == "__main__":
    #run_simulation_py()
    #run_simulation_hw()
    run_pure_math_simulation()