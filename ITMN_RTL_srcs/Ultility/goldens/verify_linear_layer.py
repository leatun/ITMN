import numpy as np

# --- CẤU HÌNH (Sửa lại cho đúng tên file của cậu) ---
INPUT_MEM = "input_linear_q3_12.mem"    # File Input X (Hex)
WEIGHT_MEM = "weight_proj1_q3_12.mem"   # File Weight W (Hex)
# BIAS_MEM = "bias_proj1.mem"           # (Optional) Nếu có bias thì uncomment
OUTPUT_CHECK_FILE = "calculated_y.mem"  # File kết quả do script này tính ra

# --- THÔNG SỐ MẠNG (Check kỹ cái này nha Zen) ---
SEQ_LEN = 1000      # Độ dài chuỗi
IN_DIM  = 64        # Đầu vào (D_MODEL)
OUT_DIM = 128       # Đầu ra (D_INNER hoặc 2*D_INNER nếu gộp) - Ở đây tớ test 1 nhánh là 128

# --- CẤU HÌNH FIXED-POINT ---
FRAC_BITS = 12
MAX_VAL = 32767
MIN_VAL = -32768

def hex_to_int(hex_str):
    """Chuyển Hex 16-bit sang Signed Integer"""
    val = int(hex_str, 16)
    if val > 32767:
        val -= 65536
    return val

def int_to_hex(val):
    """Chuyển Signed Integer sang Hex 16-bit"""
    # Clamp (Bão hòa)
    if val > MAX_VAL: val = MAX_VAL
    elif val < MIN_VAL: val = MIN_VAL
    return f"{val & 0xFFFF:04x}"

def load_mem_file(filename):
    """Đọc file .mem và trả về list các số nguyên"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(hex_to_int(line))
    return data

def main():
    print("--- STARTING SOFTWARE VERIFICATION ---")
    
    # 1. Load Data
    print(f"Loading Input: {INPUT_MEM}...")
    x_data = load_mem_file(INPUT_MEM)
    
    print(f"Loading Weight: {WEIGHT_MEM}...")
    w_data = load_mem_file(WEIGHT_MEM)
    
    # Bias (Giả sử bằng 0 nếu không có file)
    b_data = [0] * OUT_DIM
    # if BIAS_MEM: b_data = load_mem_file(BIAS_MEM) 

    # 2. Reshape (Quan trọng!)
    # Input X: [SEQ_LEN, IN_DIM]
    X = np.array(x_data).reshape(SEQ_LEN, IN_DIM)
    
    # Weight W: [OUT_DIM, IN_DIM] 
    # (Lưu ý: PyTorch Linear lưu weight theo dạng [Out, In])
    if len(w_data) != OUT_DIM * IN_DIM:
        print(f"ERROR: Kích thước Weight không khớp! Có {len(w_data)}, cần {OUT_DIM*IN_DIM}")
        return
    W = np.array(w_data).reshape(OUT_DIM, IN_DIM)

    print(f"Matrix Shapes: X{X.shape} * W.T{W.T.shape} + B{len(b_data)}")

    # 3. Tính toán (Mô phỏng phần cứng)
    # Phần cứng: Acc += (X * W) >> 0 (Full precision accumulator)
    # Sau đó: Result = (Acc >> 12) + Bias
    
    # Dùng int64 để tránh tràn số trong lúc cộng dồn
    X_long = X.astype(np.int64)
    W_long = W.astype(np.int64)
    
    # Phép nhân ma trận (MatMul)
    # Y_acc = X . W^T
    Y_acc = np.matmul(X_long, W_long.T) 
    
    # Shift và Add Bias
    # Bit shift >> 12 (Mô phỏng phép chia cho 4096)
    Y_shifted = Y_acc >> FRAC_BITS
    
    # Cộng Bias (Bias thường đã ở dạng Q3.12 nên cộng trực tiếp)
    Y_final = Y_shifted + np.array(b_data)

    # 4. Xuất file
    print(f"Saving calculated result to {OUTPUT_CHECK_FILE}...")
    with open(OUTPUT_CHECK_FILE, 'w') as f:
        for r in range(SEQ_LEN):
            for c in range(OUT_DIM):
                val = Y_final[r][c]
                f.write(int_to_hex(val) + "\n")
                
    print("--- DONE ---")
    print("Bây giờ hãy so sánh file này với file 'linear1_golden.mem' của cậu.")
    print("Gợi ý: Dùng lệnh 'diff' hoặc 'FC' trên cmd, hoặc plugin Compare trong VS Code.")

if __name__ == "__main__":
    main()