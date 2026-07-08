import numpy as np
import re

# --- CẤU HÌNH ---
D_MODEL = 64     # Input Dim (Cột)
D_INNER = 128    # Output Dim (Hàng)
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE ---
# File Weight gốc từ C++ (có index [row,col])
INPUT_FILE = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/in_proj2_weight.txt"
# File Output (Dùng để nạp vào TB Mamba System)
OUTPUT_FILE = "lin_real_w2_reordered.txt"

# --- HÀM HỖ TRỢ ---
def float_to_hex(val):
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def run():
    print(f"Reading {INPUT_FILE}...")
    
    # 1. Parse file gốc thành Ma trận (128, 64)
    # File format: [row,col] value
    w_matrix = np.zeros((D_INNER, D_MODEL))
    
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                # Regex để bắt: [row,col] value
                # Hoặc split đơn giản
                parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
                if len(parts) >= 3:
                    row = int(parts[0])
                    col = int(parts[1])
                    val = float(parts[2])
                    
                    if row < D_INNER and col < D_MODEL:
                        w_matrix[row, col] = val
                        
        print(f"Matrix Loaded. Shape: {w_matrix.shape}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. REORDERING (SẮP XẾP LẠI)
    # Mục tiêu: Tạo ra dòng chảy dữ liệu khớp với Testbench System đọc tuần tự
    # Thứ tự:
    #   Loop Chunk (0..7) -> Mỗi chunk xử lý 16 hàng
    #     Loop Time/Column (0..63) -> Mỗi nhịp cần 16 số cho 16 PE
    #       Loop Row_in_Chunk (0..15) -> Đây là 16 số nạp vào PE cùng lúc
    
    print("Reordering weights for Hardware Linear Scan...")
    
    hex_lines = []
    
    num_chunks = D_INNER // 16 # 8 chunks
    
    for chunk in range(num_chunks):
        start_row = chunk * 16
        end_row   = start_row + 16
        
        # Lấy 16 hàng của chunk này
        # Shape con: (16, 64)
        sub_matrix = w_matrix[start_row:end_row, :]
        
        # Duyệt theo CỘT (0 -> 63)
        # Tại mỗi cột, ta lấy dọc xuống 16 phần tử
        for col in range(D_MODEL):
            # Cột dọc 16 phần tử: [W[0][c], W[1][c] ... W[15][c]]
            col_data = sub_matrix[:, col]
            
            # Ghi vào danh sách
            for val in col_data:
                hex_lines.append(float_to_hex(val))

    # 3. Ghi ra file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(hex_lines))
        
    print(f"DONE! Saved to {OUTPUT_FILE}")
    print(f"Total lines: {len(hex_lines)}")
    print(f"Expected lines: {D_INNER * D_MODEL} (8192)")

if __name__ == "__main__":
    run()