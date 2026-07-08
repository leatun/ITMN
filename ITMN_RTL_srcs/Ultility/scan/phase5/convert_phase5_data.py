import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_MODEL = 64
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768


FILE_FLOAT_SCAN_OUT = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"        # Output của Scan (Input của Linear này) - Shape (1000, 128)
FILE_FLOAT_WEIGHT   = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"                # Weight Linear - Shape (64, 128)
FILE_FLOAT_GOLDEN   = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1015_15_Mixer_final_output.txt"   # Output cuối cùng - Shape (1000, 64)

# --- OUTPUT FILES (FLAT HEX) ---
FILE_FLAT_INPUT  = "phase5_input_flat.txt"
FILE_FLAT_WEIGHT = "phase5_weight_flat.txt"
FILE_FLAT_GOLDEN = "phase5_golden_flat.txt"

def to_fixed(val):
    val_fixed = int(round(val * SCALE))
    val_fixed = max(min(val_fixed, MAX_INT), MIN_INT)
    if val_fixed < 0: val_fixed += 65536
    return f"{val_fixed & 0xFFFF:04x}"

def convert_all():
    print("--- Converting Data for Phase 5 (Flat Format) ---")

    # 1. INPUT (SCAN OUT) -> RAM A
    # Pytorch: (1000 Token, 128 Channel)
    # Hardware RAM A Phase 4 lưu: Group-First.
    # Cấu trúc: [Group 0 (1000 dòng)] [Group 1 (1000 dòng)] ...
    # Mỗi dòng RAM chứa 16 kênh con.
    # Thứ tự trong file flat: Group -> Token -> Ch_in_Group (0..15)
    
    with open(FILE_FLOAT_SCAN_OUT, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    matrix_in = np.array(vals).reshape(SEQ_LEN, D_INNER)
    
    with open(FILE_FLAT_INPUT, 'w') as f:
        for g in range(8): # 8 Groups
            for t in range(SEQ_LEN): # 1000 Tokens
                # Lấy 16 kênh thuộc Group g tại thời điểm t
                start_ch = g * 16
                for k in range(16):
                    val = matrix_in[t, start_ch + k]
                    f.write(to_fixed(val) + "\n")
    print(f"1. Input Saved: {FILE_FLAT_INPUT} (128,000 lines)")

    # 2. WEIGHTS -> WEIGHT RAM
    # Pytorch: (64 Out, 128 In)
    # Hardware đọc: Chunk Output (0..3) -> Col Input (0..127) -> Row Output (0..15)
    # Thứ tự file flat: Chunk -> Col -> Row_in_Chunk (0..15)
    
    with open(FILE_FLOAT_WEIGHT, 'r') as f:
        vals = [float(x.strip().split()[-1]) if '[' in x else float(x) for x in f if x.strip()]
    W = np.array(vals).reshape(D_MODEL, D_INNER)
    
    with open(FILE_FLAT_WEIGHT, 'w') as f:
        num_chunks = D_MODEL // 16 # 4 chunks
        for chunk in range(num_chunks):
            start_row = chunk * 16
            for col in range(D_INNER):
                for r in range(16): # 16 rows trong chunk đó
                    val = W[start_row + r, col]
                    f.write(to_fixed(val) + "\n")
    print(f"2. Weight Saved: {FILE_FLAT_WEIGHT} (8,192 lines)")

    # 3. GOLDEN OUTPUT -> VERIFY
    # Pytorch: (1000 Token, 64 Out)
    # Hardware Phase 5 ghi: Token-First.
    # Cấu trúc: [Token 0] [Token 1]...
    # Mỗi Token gồm 4 Chunks (64 kênh). Mỗi Chunk 16 kênh.
    # Thứ tự file flat: Token -> Chunk -> Ch_in_Chunk (0..15)
    
    with open(FILE_FLOAT_GOLDEN, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    matrix_out = np.array(vals).reshape(SEQ_LEN, D_MODEL)
    
    with open(FILE_FLAT_GOLDEN, 'w') as f:
        for t in range(SEQ_LEN):
            for chunk in range(4): # 4 Chunks output
                start_ch = chunk * 16
                for k in range(16):
                    val = matrix_out[t, start_ch + k]
                    f.write(to_fixed(val) + "\n")
    print(f"3. Golden Saved: {FILE_FLAT_GOLDEN} (64,000 lines)")

if __name__ == "__main__":
    convert_all()