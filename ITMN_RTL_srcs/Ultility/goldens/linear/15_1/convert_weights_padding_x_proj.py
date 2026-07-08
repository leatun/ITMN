import numpy as np
import re
import math

# --- CẤU HÌNH CHO X_PROJ ---
# Input Dim = 128 (Do x_prime từ Conv ra có 128 kênh)
IN_DIM = 128     
# Output Dim = 36 (dt_rank + 2*d_state = 4 + 32 = 36)
OUT_DIM = 36    

FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE ---
INPUT_FILE = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/x_proj_weight.txt"
OUTPUT_FILE = "x_proj_weight_ptb.txt"

# --- HÀM HỖ TRỢ ---
def float_to_hex(val):
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def run():
    print(f"Reading {INPUT_FILE}...")
    print(f"Config: Input Dim={IN_DIM}, Output Dim={OUT_DIM}")
    
    # 1. Parse file gốc thành Ma trận (36, 128)
    w_matrix = np.zeros((OUT_DIM, IN_DIM))
    
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                # Format: [row,col] value
                parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
                if len(parts) >= 3:
                    row = int(parts[0])
                    col = int(parts[1])
                    val = float(parts[2])
                    
                    if row < OUT_DIM and col < IN_DIM:
                        w_matrix[row, col] = val
                        
        print(f"Matrix Loaded. Shape: {w_matrix.shape}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. REORDERING VỚI PADDING
    # Hardware đọc: 16 số (1 dòng Hex) cho 16 PE cùng lúc.
    # Thứ tự Loop: Chunk -> Col (Input) -> Row (Output 16 lines)
    
    print("Reordering weights for Hardware...")
    
    hex_lines = []
    
    # Tính số lượng Chunk (Làm tròn lên)
    # Ví dụ: 36 / 16 = 2.25 -> 3 Chunks
    num_chunks = math.ceil(OUT_DIM / 16)
    print(f"Total Chunks needed: {num_chunks}")
    
    for chunk in range(num_chunks):
        start_row = chunk * 16
        
        # Duyệt hết chiều dài Input (128 cột)
        for col in range(IN_DIM):
            
            # Tại mỗi cột, lấy 16 giá trị dọc xuống (cho 16 PE)
            for i in range(16):
                current_row = start_row + i
                
                if current_row < OUT_DIM:
                    # Nếu hàng nằm trong phạm vi (ví dụ hàng 32, 33, 34, 35)
                    val = w_matrix[current_row, col]
                else:
                    # Nếu hàng vượt quá (ví dụ hàng 36..47 của chunk cuối) -> Padding 0
                    val = 0.0
                
                # Thêm vào danh sách hex
                hex_lines.append(float_to_hex(val))

    # 3. Ghi ra file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(hex_lines))
        
    # Tính toán kích thước file mong đợi
    # 3 Chunks * 128 Cột * 16 dòng = 6144 dòng Hex (nếu file hex bung flat)
    # Hoặc nếu cậu pack 256-bit thì chia 16.
    # Script này đang xuất từng dòng 16-bit (như cậu làm trước đây).
    # Controller sẽ đọc 16 dòng này để gom thành 1 dòng 256-bit (hoặc RAM Weight cấu hình 256 bit).
    # -> Nếu RAM Weight cấu hình 256 bit, cậu cần sửa script để gom 16 số thành 1 dòng dài.
    
    # KHOAN! BRAM_256b của cậu nhận input 256 bit (dma_wdata).
    # Nhưng script cũ cậu dùng (in_proj2) xuất từng dòng hex 16-bit.
    # Trong TB cậu viết: "dma_wdata[k*16 +: 16] = file_w[i*16 + k]"
    # -> Tức là TB tự gom. Vậy script xuất ra từng dòng 16-bit là ĐÚNG.
    
    expected_lines = num_chunks * IN_DIM * 16
    print(f"DONE! Saved to {OUTPUT_FILE}")
    print(f"Total hex lines: {len(hex_lines)}")
    print(f"Expected lines: {expected_lines}")

if __name__ == "__main__":
    run()