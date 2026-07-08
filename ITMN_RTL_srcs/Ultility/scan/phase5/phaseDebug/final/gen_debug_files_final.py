import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128  # Input Dimension
D_MODEL = 64   # Output Dimension
FRAC_BITS = 11
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE RAW (FLOAT) ---
# Dùng đúng file mà cậu vừa Verify Math thành công
FILE_IN_RAW  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_W_RAW   = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
FILE_OUT_RAW = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"

# --- OUTPUT FILES (HEX FLAT) ---
OUT_HEX_INPUT  = "debug_input_correct.txt"
OUT_HEX_WEIGHT = "debug_weight_correct.txt"
OUT_HEX_GOLDEN = "debug_golden_correct.txt"

def to_fixed(val_float):
    val = int(round(val_float * SCALE))
    val = max(min(val, MAX_INT), MIN_INT)
    if val < 0: val += 65536
    return f"{val & 0xFFFF:04x}"

def run_conversion():
    print("--- 1. CONVERTING INPUT (Transpose Logic) ---")
    with open(FILE_IN_RAW, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    
    # Load vào với Shape gốc từ file Raw
    raw_matrix = np.array(vals).reshape(D_INNER, SEQ_LEN) # (128, 1000)
    print(f"   Raw Shape: {raw_matrix.shape}")
    
    # TRANSPOSE để thành Token-First (1000, 128) cho Hardware dễ đọc tuần tự
    input_matrix = raw_matrix.T 
    print(f"   Transposed Shape: {input_matrix.shape} (Token, Channel)")
    
    with open(OUT_HEX_INPUT, 'w') as f:
        for t in range(SEQ_LEN):
            for k in range(D_INNER):
                f.write(to_fixed(input_matrix[t, k]) + "\n")
    print(f"   -> Saved {OUT_HEX_INPUT}")

    print("\n--- 2. CONVERTING WEIGHT (Reorder Logic) ---")
    w_vals = []
    with open(FILE_W_RAW, 'r') as f:
        for line in f:
            parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
            if len(parts) >= 3: w_vals.append(float(parts[2]))
            else: w_vals.append(float(parts[0]))
            
    W = np.array(w_vals).reshape(D_MODEL, D_INNER) # (64, 128)
    print(f"   Weight Shape: {W.shape}")
    
    # Reorder: Chunk Output (0..3) -> Col Input (0..127) -> Row Output (0..15)
    with open(OUT_HEX_WEIGHT, 'w') as f:
        num_chunks = D_MODEL // 16
        for chunk in range(num_chunks):
            start_row = chunk * 16
            for col in range(D_INNER):
                for r in range(16):
                    val = W[start_row + r, col]
                    f.write(to_fixed(val) + "\n")
    print(f"   -> Saved {OUT_HEX_WEIGHT}")

    print("\n--- 3. CONVERTING GOLDEN OUTPUT ---")
    with open(FILE_OUT_RAW, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    
    # Golden Output gốc là Token-First (1000, 64) -> Không cần Transpose
    gold_matrix = np.array(vals).reshape(SEQ_LEN, D_MODEL)
    print(f"   Golden Shape: {gold_matrix.shape}")
    
    with open(OUT_HEX_GOLDEN, 'w') as f:
        for t in range(SEQ_LEN):
            for k in range(D_MODEL):
                f.write(to_fixed(gold_matrix[t, k]) + "\n")
    print(f"   -> Saved {OUT_HEX_GOLDEN}")

if __name__ == "__main__":
    run_conversion()