import numpy as np

# --- CẤU HÌNH ---

# INPUT_FILE = "cpp_golden_files/07_07_MambaBlock_after_norm.txt" # Tên file dữ liệu
# OUTPUT_FILE = "input_linear_q3_12.mem"         # Tên file Hex xuất ra

INPUT_FILE = "cpp_golden_files/07_07_MambaBlock_after_norm.txt" # Tên file dữ liệu của cậu
OUTPUT_FILE = "linear_x_input.mem"         # Tên file Hex xuất ra

FRAC_BITS = 12                                 # Q3.12
SCALE = 1 << FRAC_BITS                         # 4096
MAX_VAL = 32767                                # Giới hạn dương 16-bit
MIN_VAL = -32768                               # Giới hạn âm 16-bit

def float_to_hex_q3_12(val):
    # 1. Scale
    scaled = val * SCALE
    
    # 2. Round (Làm tròn)
    int_val = int(round(scaled))
    
    # 3. Clamp (Kẹp giá trị để không tràn số)
    if int_val > MAX_VAL:
        int_val = MAX_VAL
    elif int_val < MIN_VAL:
        int_val = MIN_VAL
        
    # 4. Convert to Hex (Two's Complement cho số âm)
    # Phép & 0xFFFF sẽ tự động chuyển số âm -1 thành FFFF
    return f"{int_val & 0xFFFF:04x}"

def main():
    print(f"Loading data from {INPUT_FILE}...")
    
    try:
        # Dùng numpy load cho nhanh, nó tự xử lý space và scientific notation (e-01)
        data = np.loadtxt(INPUT_FILE)
        
        # Check kích thước
        rows, cols = data.shape
        print(f"Shape detected: {rows} rows (Seq Len), {cols} columns (Dim)")
        
        if cols != 64:
            print("WARNING: Số cột không phải 64? Kiểm tra lại parameter phần cứng!")

        print(f"Converting to Q3.{FRAC_BITS} Hex format...")
        
        with open(OUTPUT_FILE, "w") as f:
            # Duyệt từng dòng (Sequence step)
            for r in range(rows):
                # Duyệt từng cột (Feature dim)
                # Phần cứng thường đọc tuyến tính: x[0], x[1]... x[63] của token 0
                # Sau đó đến x[0]...x[63] của token 1
                for c in range(cols):
                    float_val = data[r][c]
                    hex_str = float_to_hex_q3_12(float_val)
                    
                    # Ghi mỗi giá trị 1 dòng (để $readmemh đọc)
                    f.write(hex_str + "\n")
                    
        print(f"Success! Output saved to {OUTPUT_FILE}")
        print("Example conversion:")
        print(f"Float: {data[0][0]} -> Hex: {float_to_hex_q3_12(data[0][0])}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()