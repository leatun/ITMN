import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
SCALE = 1 << 12
MAX_INT = 32767
MIN_INT = -32768

INPUT_FILE = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt" # File float gốc
OUTPUT_FILE = "debug_input_token_first.txt"

def to_fixed(val):
    val = int(round(val * SCALE))
    val = max(min(val, MAX_INT), MIN_INT)
    if val < 0: val += 65536
    return f"{val & 0xFFFF:04x}"

def run():
    print("Generating Token-First Input for Debug...")
    with open(INPUT_FILE, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    
    # Shape gốc (1000, 128). Flatten thẳng tuột là ra Token-First
    matrix = np.array(vals).reshape(SEQ_LEN, D_INNER)
    
    with open(OUTPUT_FILE, 'w') as f:
        for t in range(SEQ_LEN):
            for k in range(D_INNER):
                f.write(to_fixed(matrix[t, k]) + "\n")
                
    print(f"Done. Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    run()