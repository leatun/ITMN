import numpy as np

# --- CẤU HÌNH ---
INPUT_FILE = "cpp_golden_files/08_X_after_linear.txt"  # File gốc (Shape 1000 x 256)
OUTPUT_FILE_1 = "linear1_golden.mem" # File kết quả 1 (Cột 0-127)
OUTPUT_FILE_2 = "linear2_golden.mem" # File kết quả 2 (Cột 128-255)

# Cấu hình Fixed-Point Q3.12
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_VAL = 32767
MIN_VAL = -32768

def float_to_hex_q3_12(val):
    """Chuyển float sang Hex 16-bit Q3.12"""
    try:
        scaled = val * SCALE
        int_val = int(round(scaled))
        if int_val > MAX_VAL: int_val = MAX_VAL
        elif int_val < MIN_VAL: int_val = MIN_VAL
        return f"{int_val & 0xFFFF:04x}"
    except:
        return "0000"

def save_mem_file(data_matrix, filename):
    """Hàm phụ trợ để ghi ma trận ra file .mem"""
    rows, cols = data_matrix.shape
    print(f"   -> Saving {filename} (Shape: {rows}x{cols})...")
    
    with open(filename, 'w') as f:
        for r in range(rows):
            for c in range(cols):
                hex_str = float_to_hex_q3_12(data_matrix[r][c])
                f.write(hex_str + "\n")
    print(f"   -> Done: {filename}")

def process_and_split(in_path, out1_path, out2_path):
    print(f"\n--- Processing & Splitting: {in_path} ---")
    try:
        # Load dữ liệu
        data = np.loadtxt(in_path)
        rows, cols = data.shape
        print(f"Original Shape: {rows} rows, {cols} columns")
        
        # Kiểm tra xem có đủ để cắt đôi không
        if cols < 256:
            print("WARNING: Số cột nhỏ hơn 256! Kiểm tra lại file gốc.")
        
        # --- CẮT ĐÔI (Slicing) ---
        # Part 1: Lấy từ cột 0 đến 127
        part1 = data[:, :128]
        
        # Part 2: Lấy từ cột 128 đến hết (hoặc 255)
        part2 = data[:, 128:]
        
        # --- GHI FILE ---
        save_mem_file(part1, out1_path)
        save_mem_file(part2, out2_path)
        
        print("\nSUCCESS! Split complete.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Chạy hàm xử lý
    process_and_split(INPUT_FILE, OUTPUT_FILE_1, OUTPUT_FILE_2)