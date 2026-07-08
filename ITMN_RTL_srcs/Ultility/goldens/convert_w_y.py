import numpy as np
import re

# --- CẤU HÌNH ---
# 1. File Weight (Có dính index [0,0])
# WEIGHT_INPUT_FILE = "golden_vectors_txt/in_proj1_weight.txt"
# WEIGHT_OUTPUT_FILE = "weight_proj1_q3_12.mem"

WEIGHT_INPUT_FILE = "golden_vectors_txt/in_proj2_weight.txt"
WEIGHT_OUTPUT_FILE = "linear_weight_proj2.mem"

# # 2. File Golden Output (Ma trận thuần số)
# GOLDEN_INPUT_FILE = "cpp_golden_files/08_X_after_linear.txt"
# GOLDEN_OUTPUT_FILE = "y_golden_q3_12.mem"

# 3. Cấu hình Fixed-Point
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_VAL = 32767
MIN_VAL = -32768

def float_to_hex_q3_12(val):
    """Chuyển float sang Hex 16-bit Q3.12 (có kẹp dòng/clamp)"""
    try:
        scaled = val * SCALE
        int_val = int(round(scaled))
        if int_val > MAX_VAL: int_val = MAX_VAL
        elif int_val < MIN_VAL: int_val = MIN_VAL
        return f"{int_val & 0xFFFF:04x}"
    except:
        return "0000" # Fallback nếu lỗi

def process_weight_file(in_path, out_path):
    print(f"\n--- Processing Weight File: {in_path} ---")
    count = 0
    try:
        with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
            for line in fin:
                # Line mẫu: "  [0,0] 0.039038"
                # Cách xử lý: Tách chuỗi bằng khoảng trắng, lấy phần tử cuối cùng
                parts = line.strip().split()
                if not parts: continue
                
                # Phần tử cuối cùng là giá trị float
                str_val = parts[-1] 
                
                try:
                    float_val = float(str_val)
                    hex_str = float_to_hex_q3_12(float_val)
                    fout.write(hex_str + "\n")
                    count += 1
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")
                    
        print(f"Done! Converted {count} weights to {out_path}")
    except FileNotFoundError:
        print(f"Error: File {in_path} not found!")

def process_matrix_file(in_path, out_path):
    print(f"\n--- Processing Golden Matrix: {in_path} ---")
    try:
        # Dùng numpy load ma trận thuần số (nhanh hơn tự parse)
        data = np.loadtxt(in_path)
        
        # Nếu file chỉ có 1 dòng hoặc 1 cột, numpy có thể load ra vector 1D
        if data.ndim == 1:
            rows, cols = data.shape[0], 1
            data = data.reshape(-1, 1)
        else:
            rows, cols = data.shape
            
        print(f"Shape detected: {rows} x {cols}")
        
        with open(out_path, 'w') as f:
            for r in range(rows):
                for c in range(cols):
                    float_val = data[r][c]
                    hex_str = float_to_hex_q3_12(float_val)
                    f.write(hex_str + "\n")
                    
        print(f"Done! Converted matrix to {out_path}")
    except Exception as e:
        print(f"Error processing matrix file: {e}")

if __name__ == "__main__":
    # 1. Convert Weight (Xử lý cái file có [0,0])
    process_weight_file(WEIGHT_INPUT_FILE, WEIGHT_OUTPUT_FILE)
    
    # 2. Convert Golden Output (Xử lý file ma trận thuần)
    # Nhớ đổi tên file input cho đúng
    #process_matrix_file(GOLDEN_INPUT_FILE, GOLDEN_OUTPUT_FILE) 
    # (Bỏ comment dòng trên nếu cậu đã có file golden)