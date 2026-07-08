import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128  # Input Dimension
D_MODEL = 64   # Output Dimension
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE FLOAT GỐC (Sửa lại đường dẫn của cậu) ---
FILE_FLOAT_INPUT  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_FLOAT_WEIGHT = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
FILE_FLOAT_GOLDEN = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"

# --- OUTPUT FILES (HEX FLAT) ---
OUT_HEX_INPUT  = "debug_input_token_first1.txt"
OUT_HEX_WEIGHT = "debug_weight_reordered.txt"
OUT_HEX_GOLDEN = "debug_golden_token_first.txt"

def to_fixed(val):
    val_fixed = int(round(val * SCALE))
    val_fixed = max(min(val_fixed, MAX_INT), MIN_INT)
    if val_fixed < 0: val_fixed += 65536
    return f"{val_fixed & 0xFFFF:04x}"

def gen_input():
    print(f"1. Generating Input (Token-First) -> {OUT_HEX_INPUT}")
    # Đọc file float (1000, 128)
    with open(FILE_FLOAT_INPUT, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    matrix = np.array(vals).reshape(SEQ_LEN, D_INNER)
    
    # Ghi file Flat: Token 0 (128 số) -> Token 1 (128 số)...
    # Hardware đọc tuần tự: Dòng 0 (Ch0-15), Dòng 1 (Ch16-31)...
    with open(OUT_HEX_INPUT, 'w') as f:
        for t in range(SEQ_LEN):
            for k in range(D_INNER):
                f.write(to_fixed(matrix[t, k]) + "\n")

def gen_weight():
    print(f"2. Generating Weight (Chunk-First) -> {OUT_HEX_WEIGHT}")
    # Đọc file float (64, 128) -> (Out, In)
    with open(FILE_FLOAT_WEIGHT, 'r') as f:
        # Xử lý format [row, col] val hoặc just val
        vals = []
        for line in f:
            parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
            if len(parts) >= 3: vals.append(float(parts[2])) # Format [r,c] val
            else: vals.append(float(parts[0])) # Format val only
            
    W = np.array(vals).reshape(D_MODEL, D_INNER)
    
    # Reorder: Loop Chunk Output (0..3) -> Loop Input Col (0..127) -> Row (16)
    with open(OUT_HEX_WEIGHT, 'w') as f:
        num_chunks = D_MODEL // 16 # 4 chunks
        for chunk in range(num_chunks):
            start_row = chunk * 16
            for col in range(D_INNER):
                for r in range(16): # 16 rows song song
                    val = W[start_row + r, col]
                    f.write(to_fixed(val) + "\n")

def gen_golden():
    print(f"3. Generating Golden (Token-First) -> {OUT_HEX_GOLDEN}")
    # Đọc file float (1000, 64)
    with open(FILE_FLOAT_GOLDEN, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    matrix = np.array(vals).reshape(SEQ_LEN, D_MODEL)
    
    # Ghi file Flat: Token 0 (64 số) -> Token 1...
    with open(OUT_HEX_GOLDEN, 'w') as f:
        for t in range(SEQ_LEN):
            for k in range(D_MODEL):
                f.write(to_fixed(matrix[t, k]) + "\n")

if __name__ == "__main__":
    gen_input()
    gen_weight()
    gen_golden()
    print("--- ALL DONE ---")