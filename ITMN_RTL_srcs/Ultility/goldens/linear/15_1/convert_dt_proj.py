import numpy as np
import math

# --- CẤU HÌNH ---
# Dt_Proj: Input=4 (dt_rank), Output=128 (d_inner)
IN_DIM = 4      
OUT_DIM = 128   

FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE ---
FILE_W_IN = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/dt_proj_weight.txt"
FILE_B_IN = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/dt_proj_bias.txt"

FILE_W_OUT = "dt_proj_weight_ptb.txt"
FILE_B_OUT = "dt_proj_bias_ptb.txt"

def float_to_hex(val):
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def process_weights():
    print(f"--- Processing Weights (Shape {OUT_DIM},{IN_DIM}) ---")
    w_matrix = np.zeros((OUT_DIM, IN_DIM))
    try:
        with open(FILE_W_IN, 'r') as f:
            for line in f:
                parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
                if len(parts) >= 3:
                    row, col, val = int(parts[0]), int(parts[1]), float(parts[2])
                    if row < OUT_DIM and col < IN_DIM:
                        w_matrix[row, col] = val
    except Exception as e:
        print(f"Error W: {e}"); return

    # Reorder: Chunk -> Col -> Row (16)
    hex_lines = []
    num_chunks = math.ceil(OUT_DIM / 16) # 128/16 = 8 chunks
    
    for chunk in range(num_chunks):
        start_row = chunk * 16
        for col in range(IN_DIM):
            for i in range(16):
                r = start_row + i
                val = w_matrix[r, col] if r < OUT_DIM else 0.0
                hex_lines.append(float_to_hex(val))
                
    with open(FILE_W_OUT, 'w') as f:
        f.write('\n'.join(hex_lines))
    print(f"Saved Weights to {FILE_W_OUT}")

def process_bias():
    print(f"--- Processing Bias (Shape {OUT_DIM}) ---")
    b_vec = []
    try:
        with open(FILE_B_IN, 'r') as f:
            for line in f:
                # Bias file chỉ có số float mỗi dòng
                val = float(line.strip())
                b_vec.append(val)
    except Exception as e:
        print(f"Error B: {e}"); return

    # Padding nếu thiếu
    while len(b_vec) < OUT_DIM: b_vec.append(0.0)
    
    # Reorder: Thực ra Bias lưu tuần tự 128 số là khớp với thứ tự Chunk rồi
    # Hardware đọc 16 số một lần -> Chính là 16 số liên tiếp trong file này.
    hex_lines = [float_to_hex(x) for x in b_vec]
    
    with open(FILE_B_OUT, 'w') as f:
        f.write('\n'.join(hex_lines))
    print(f"Saved Bias to {FILE_B_OUT}")

if __name__ == "__main__":
    process_weights()
    process_bias()