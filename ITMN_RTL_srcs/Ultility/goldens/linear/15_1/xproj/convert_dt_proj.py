import numpy as np
import math
import re

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
# Dt_Proj: Input=4 (dt_rank), Output=128 (d_inner)
IN_DIM = 4      
OUT_DIM = 128   

FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE (Sửa lại đường dẫn của cậu) ---
DIR = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"
FILE_W_IN = DIR + "dt_proj_weight.txt" # Format: [row,col] value
FILE_B_IN = DIR + "dt_proj_bias.txt"   # Format: value (hoặc [idx] value)

FILE_W_OUT = "w_dt_proj_reordered.txt"
FILE_B_OUT = "b_dt_proj.txt"

# ==============================================================================
# HÀM XỬ LÝ
# ==============================================================================
def float_to_hex(val):
    """Convert float -> Q3.12 Hex String"""
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def parse_weight_file(filepath, rows, cols):
    """Đọc file dạng [r,c] val vào Matrix"""
    matrix = np.zeros((rows, cols))
    print(f"Reading Weights from {filepath}...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Dùng Regex để bắt số bất kể format [r, c] hay [r,c]
            # Tìm tất cả các số (bao gồm cả dấu chấm và dấu âm)
            # Ví dụ: [0,1] -0.05 -> matches: 0, 1, -0.05
            parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            
            if len(parts) >= 3:
                r = int(parts[0])
                c = int(parts[1])
                v = float(parts[2])
                
                if r < rows and c < cols:
                    matrix[r, c] = v
    return matrix

def parse_bias_file(filepath, size):
    """Đọc file Bias. Hỗ trợ cả dạng [idx] val hoặc chỉ val"""
    vec = np.zeros(size)
    print(f"Reading Bias from {filepath}...")
    
    with open(filepath, 'r') as f:
        idx_counter = 0
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            
            if len(parts) >= 2: # Dạng [idx] val
                idx = int(parts[0])
                v = float(parts[1])
                if idx < size: vec[idx] = v
            elif len(parts) == 1: # Dạng chỉ có val -> tự tăng index
                v = float(parts[0])
                if idx_counter < size: 
                    vec[idx_counter] = v
                    idx_counter += 1
    return vec

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def run():
    # 1. LOAD DATA
    W = parse_weight_file(FILE_W_IN, OUT_DIM, IN_DIM)
    B = parse_bias_file(FILE_B_IN, OUT_DIM)
    
    # 2. PROCESS WEIGHTS (REORDERING)
    # Target: 8 Chunks. Mỗi Chunk 4 Cols. Mỗi Col 16 Rows (PEs).
    hex_lines_w = []
    num_chunks = math.ceil(OUT_DIM / 16) # 8
    
    print("Reordering Weights...")
    for chunk in range(num_chunks):
        start_row = chunk * 16
        # Loop theo thứ tự phần cứng: Col (Input) -> Row (PE)
        for col in range(IN_DIM): # 0..3
            for i in range(16):   # 0..15
                r = start_row + i
                if r < OUT_DIM:
                    val = W[r, col]
                else:
                    val = 0.0 # Padding nếu OUT_DIM không chia hết cho 16
                hex_lines_w.append(float_to_hex(val))
    
    with open(FILE_W_OUT, 'w') as f:
        f.write('\n'.join(hex_lines_w))
    print(f"-> Saved Weights: {FILE_W_OUT} (Lines: {len(hex_lines_w)})")
    
    # 3. PROCESS BIAS
    # Target: Hardware đọc tuần tự mỗi lần 1 dòng 256-bit (16 số)
    # Vì Bias trong file gốc thường đã sort theo output channel index -> Chỉ cần convert thẳng.
    print("Processing Bias...")
    hex_lines_b = [float_to_hex(x) for x in B]
    
    with open(FILE_B_OUT, 'w') as f:
        f.write('\n'.join(hex_lines_b))
    print(f"-> Saved Bias: {FILE_B_OUT} (Lines: {len(hex_lines_b)})")

if __name__ == "__main__":
    run()