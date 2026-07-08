import numpy as np
import os

# ==============================================================================
# CẤU HÌNH HỆ THỐNG
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128   # Input Dimension
D_OUT   = 36    # Output Dimension (B=16, C=16, dt=4)
FRAC_BITS = 12
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE (SỬA LẠI CHO ĐÚNG ĐƯỜNG DẪN CỦA CẬU) ---
BASE_DIR = "." # Thư mục hiện tại

# 1. Input File (File Chunked mà cậu mới tạo cho TB)
F_INPUT_CHUNKED = "x_prime_chunked_for_tb.txt" # Hoặc x_prime_chunked_verified.txt

# 2. Weight File (File Hex nạp vào TB)
F_WEIGHT_XPROJ  = "w_xproj_reordered.txt"

# 3. Golden Output Files (Để so sánh)
F_GOLD_B        = "scan_real_B_shared.txt"
F_GOLD_C        = "scan_real_C_shared.txt"

# ==============================================================================
# CÁC HÀM XỬ LÝ SỐ HỌC (MÔ PHỎNG PHẦN CỨNG)
# ==============================================================================

def hex_to_signed(hex_str):
    """Chuyển chuỗi Hex 16-bit sang số nguyên có dấu (Python int)"""
    val = int(hex_str, 16)
    if val & 0x8000: # Nếu bit dấu là 1
        val = val - 0x10000
    return val

def to_hex(val):
    """Chuyển số nguyên sang Hex 16-bit (để in debug)"""
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def fixed_mac(vec_a, vec_b):
    """
    Mô phỏng phép nhân chập Linear Layer (MAC).
    Input: 2 vector số nguyên (đã convert từ hex).
    Logic: Sum(A * B) >> FRAC_BITS
    """
    accumulator = 0
    
    # 1. Multiply & Accumulate (Full Precision)
    # Phần cứng thực tế dùng DSP slice, tích lũy trong thanh ghi 48-bit
    for a, b in zip(vec_a, vec_b):
        accumulator += (a * b)
        
    # 2. Shift Right (Quantization)
    # Lưu ý: Python shift số âm hơi khác Verilog một chút ở cách làm tròn,
    # nhưng với >> thuần túy thì thường khớp.
    res = accumulator >> FRAC_BITS
    
    # 3. Saturation (Kẹp giá trị về 16-bit)
    if res > MAX_INT: res = MAX_INT
    elif res < MIN_INT: res = MIN_INT
    
    return res

# ==============================================================================
# HÀM LOAD DỮ LIỆU
# ==============================================================================

def load_input_chunked(filepath):
    """
    Đọc file input dạng Chunked (Channel-Grouped) và tái tạo lại ma trận Input.
    File format: Loop Chunk(0..7) -> Loop Token(0..999) -> Loop 16 Channels.
    Output mong muốn: Matrix [1000][128] (Token-First để dễ tính Linear)
    """
    print(f"Loading Input from {filepath}...")
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    # Total lines = 128 * 1000 = 128000
    expected_lines = D_INNER * SEQ_LEN
    if len(lines) != expected_lines:
        print(f"Warning: File lines {len(lines)} != Expected {expected_lines}")

    # Reconstruct Matrix X [Token][Channel]
    # Logic: 
    # Chunk 0 (lines 0 -> 15999) chứa Ch 0-15.
    # Trong Chunk 0: 16 dòng đầu là Token 0, 16 dòng tiếp là Token 1...
    
    X_matrix = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    
    lines_per_chunk = SEQ_LEN * 16
    
    for chunk in range(8): # 8 chunks
        chunk_offset = chunk * lines_per_chunk
        base_ch = chunk * 16
        
        for t in range(SEQ_LEN):
            token_offset = t * 16
            # Đọc 16 kênh của token t trong chunk này
            for i in range(16):
                line_idx = chunk_offset + token_offset + i
                val = hex_to_signed(lines[line_idx])
                
                # Gán vào ma trận
                X_matrix[t][base_ch + i] = val
                
    print("-> Input Matrix Reconstructed.")
    return X_matrix

def load_weights(filepath):
    """
    Đọc file Weight (Format Hex nạp cho TB).
    File này thường là Token-First hoặc Chunk-based tùy script tạo.
    Script convert_weights_padding.py tạo ra: Loop Chunk -> Loop Col -> Loop 16 Rows.
    Ta cần tái tạo thành Matrix W [Output=36][Input=128]
    """
    print(f"Loading Weights from {filepath}...")
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    # Logic script tạo weight: 
    # 3 Chunks. Mỗi Chunk duyệt 128 Cột. Mỗi Cột có 16 Hàng.
    # Total lines = 3 * 128 * 16 = 6144.
    
    W_matrix = np.zeros((48, 128), dtype=int) # 48 rows (do padding), 128 cols
    
    idx = 0
    for chunk in range(3):
        start_row = chunk * 16
        for col in range(128):
            for i in range(16): # 16 rows trong 1 cột
                if idx < len(lines):
                    val = hex_to_signed(lines[idx])
                    W_matrix[start_row + i][col] = val
                    idx += 1
                    
    # Cắt bỏ phần padding, chỉ lấy 36 dòng thực
    # W_matrix = W_matrix[:36, :] 
    # Tuy nhiên, để mô phỏng đúng phần cứng, ta cứ để 48 dòng, 
    # rồi lúc lấy kết quả chỉ lấy 36 cái đầu.
    
    print("-> Weight Matrix Loaded.")
    return W_matrix

def load_golden(filepath):
    print(f"Loading Golden from {filepath}...")
    with open(filepath, 'r') as f:
        # File golden lưu theo Token-First: T0(16nums), T1(16nums)...
        vals = [hex_to_signed(line.strip()) for line in f if line.strip()]
    return np.array(vals).reshape(SEQ_LEN, 16) # [Token][16]

# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def run_simulation():
    # 1. Load Data
    X_matrix = load_input_chunked(F_INPUT_CHUNKED)
    W_matrix = load_weights(F_WEIGHT_XPROJ)
    
    Gold_B = load_golden(F_GOLD_B)
    Gold_C = load_golden(F_GOLD_C)
    
    print("\n--- STARTING PYTHON HARDWARE SIMULATION ---")
    
    err_B_count = 0
    err_C_count = 0
    
    # 2. Loop Tokens (Mô phỏng từng Token như Hardware)
    for t in range(SEQ_LEN):
        # Lấy vector input của token t (128 phần tử)
        x_vec = X_matrix[t]
        
        # Tính Linear: Y = W * X
        # Hardware tính 3 chunks.
        # Chunk 0: Rows 0-15 (Tương ứng B)
        # Chunk 1: Rows 16-31 (Tương ứng C)
        # Chunk 2: Rows 32-47 (Có dt ở đầu)
        
        # --- Calc B (Rows 0-15) ---
        sim_B = []
        for r in range(16):
            w_row = W_matrix[r]
            res = fixed_mac(x_vec, w_row)
            sim_B.append(res)
            
        # --- Calc C (Rows 16-31) ---
        sim_C = []
        for r in range(16, 32):
            w_row = W_matrix[r]
            res = fixed_mac(x_vec, w_row)
            sim_C.append(res)
            
        # --- Calc dt_raw (Rows 32-35) ---
        sim_dt = []
        for r in range(32, 36):
            w_row = W_matrix[r]
            res = fixed_mac(x_vec, w_row)
            sim_dt.append(res)
            
        # 3. Compare with Golden
        # So sánh B
        gold_b_vec = Gold_B[t]
        for i in range(16):
            diff = abs(sim_B[i] - gold_b_vec[i])
            if diff > 15: # Tolerance
                if err_B_count < 5:
                    print(f"[ERR B] Tok {t} Idx {i} | Sim: {to_hex(sim_B[i])} Gold: {to_hex(gold_b_vec[i])} Diff: {diff}")
                err_B_count += 1
                
        # So sánh C
        gold_c_vec = Gold_C[t]
        for i in range(16):
            diff = abs(sim_C[i] - gold_c_vec[i])
            if diff > 15:
                if err_C_count < 5:
                    print(f"[ERR C] Tok {t} Idx {i} | Sim: {to_hex(sim_C[i])} Gold: {to_hex(gold_c_vec[i])} Diff: {diff}")
                err_C_count += 1

    print("\n--- SIMULATION RESULT ---")
    print(f"Total B Errors: {err_B_count}")
    print(f"Total C Errors: {err_C_count}")
    
    if err_B_count == 0 and err_C_count == 0:
        print(">>> PYTHON SIM MATCHES GOLDEN FILE! <<<")
        print("Điều này chứng tỏ:")
        print("1. File Input (Chunked) đã được tạo ĐÚNG logic.")
        print("2. File Weight đã được tạo ĐÚNG logic.")
        print("3. File Golden B/C là ĐÚNG.")
        print("=> Nếu Verilog vẫn sai, thì lỗi nằm ở Verilog Controller.")
    else:
        print(">>> PYTHON SIM MISMATCH! <<<")
        print("Điều này chứng tỏ Input, Weight hoặc Logic tính toán đang bị lệch pha.")
        print("Hãy kiểm tra lại file input và weight.")

if __name__ == "__main__":
    run_simulation()