import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_MODEL = 64
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE GỐC (FLOAT) ---
# Dùng đúng file cậu đang có
FILE_FLOAT_INPUT  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/1014_14_Mixer_y_gated.txt"
FILE_FLOAT_WEIGHT = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"

# --- OUTPUT ---
# File này sẽ được dùng để thay thế file cũ trong Testbench
OUTPUT_HEX_FILE   = "phase5_golden_flat.txt" 

# --- HÀM MÔ PHỎNG HARDWARE ---
def to_fixed(val_float):
    val = int(round(val_float * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    return val

def to_hex(val_int):
    if val_int < 0: val_int += 65536
    return f"{val_int & 0xFFFF:04x}"

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def run_simulation():
    print("--- GENERATING HARDWARE-MATCHING GOLDEN ---")
    
    # 1. Load và Quantize Input
    print(f"Reading Input: {FILE_FLOAT_INPUT}")
    with open(FILE_FLOAT_INPUT, 'r') as f:
        vals = [float(x) for x in f.read().split()]
    # Chuyển sang Fixed-point ngay từ đầu
    X = np.array([to_fixed(x) for x in vals]).reshape(SEQ_LEN, D_INNER)
    
    # 2. Load và Quantize Weight
    print(f"Reading Weight: {FILE_FLOAT_WEIGHT}")
    w_vals = []
    with open(FILE_FLOAT_WEIGHT, 'r') as f:
        for line in f:
            parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
            if len(parts) >= 3: w_vals.append(float(parts[2]))
            else: w_vals.append(float(parts[0]))
    W = np.array([to_fixed(x) for x in w_vals]).reshape(D_MODEL, D_INNER)

    # 3. Tính toán Linear theo logic "Nhà nghèo" (Shift-then-Add) - KHÔNG BIAS
    print("Calculating Linear (Hardware Logic: (A*B)>>12, No Bias)...")
    
    Y_HW = np.zeros((SEQ_LEN, D_MODEL), dtype=int)
    
    for t in range(SEQ_LEN):
        for out_ch in range(D_MODEL):
            acc = 0
            # Dot Product
            for in_ch in range(D_INNER):
                # Mô phỏng PE: Nhân -> Dịch bit -> Cộng dồn
                prod = int(X[t, in_ch]) * int(W[out_ch, in_ch])
                term = prod >> FRAC_BITS # Arithmetic Shift
                acc += term
                # acc = sat16(acc) # Uncomment nếu PE có kẹp sau mỗi lần cộng
            
            # Kẹp đầu ra cuối cùng
            Y_HW[t, out_ch] = sat16(acc)
            
        if t % 200 == 0: print(f"  Processed {t} tokens...")

    # 4. Xuất file Hex Flat (Token-First)
    print(f"Saving to {OUTPUT_HEX_FILE}...")
    with open(OUTPUT_HEX_FILE, 'w') as f:
        for t in range(SEQ_LEN):
            # Token-First: Ghi hết 64 kênh của token t
            for chunk in range(4): # 4 Chunks
                start_ch = chunk * 16
                for k in range(16):
                    val = Y_HW[t, start_ch + k]
                    f.write(to_hex(val) + "\n")
                    
    print("--- DONE! ---")
    print("Bây giờ hãy chạy lại Testbench 'tb_Phase5_Flat'. Nó sẽ dùng file này để so sánh.")
    print("Nếu Hardware đúng, nó sẽ PASS PERFECTLY (hoặc sai số cực nhỏ 1-2 unit).")

if __name__ == "__main__":
    run_simulation()