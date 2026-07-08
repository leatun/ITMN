import numpy as np
import re

# --- CẤU HÌNH CHO PHASE 5 (OUT PROJ) ---
# Phase 1: In=64, Out=128
# Phase 5: In=128, Out=64 (NGƯỢC LẠI)

D_MODEL = 64     # Output Dim (Hàng của ma trận W) - Cần chia chunk theo cái này
D_INNER = 128    # Input Dim (Cột của ma trận W)   - Cần duyệt hết cái này mới xong 1 chunk
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE ---
INPUT_FILE = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/out_proj_weight.txt"
OUTPUT_FILE = "phase5_weight_reordered.txt"

# --- HÀM HỖ TRỢ ---
def float_to_hex(val):
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def run():
    print(f"Reading {INPUT_FILE} for Phase 5...")
    
    # 1. Parse file gốc thành Ma trận
    # Pytorch Linear: y = xA^T + b.
    # Weight shape trong Pytorch là (Out_Features, In_Features) = (64, 128)
    # File text thường lưu dạng: [row, col] value -> row là Out Channel, col là In Channel.
    
    w_matrix = np.zeros((D_MODEL, D_INNER)) # (64, 128)
    
    try:
        with open(INPUT_FILE, 'r') as f:
            for line in f:
                parts = line.strip().replace('[', '').replace(']', '').replace(',', ' ').split()
                if len(parts) >= 3:
                    row = int(parts[0])   # Output Channel Index
                    col = int(parts[1])   # Input Channel Index
                    val = float(parts[2])
                    
                    if row < D_MODEL and col < D_INNER:
                        w_matrix[row, col] = val
                        
        print(f"Matrix Loaded. Shape: {w_matrix.shape}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. REORDERING (SẮP XẾP LẠI CHO PHASE 5)
    # Thứ tự Hardware Phase 5 cần:
    #   Loop Output Chunk (0..3) -> (64 out / 16 pe = 4 chunks)
    #     Loop Input Time/Col (0..127) -> (128 input channels)
    #       Loop Row_in_Chunk (0..15) -> 16 PE song song
    
    print("Reordering weights for Phase 5 Output Projection...")
    
    hex_lines = []
    
    num_chunks = D_MODEL // 16 # 64 / 16 = 4 chunks
    
    for chunk in range(num_chunks):
        start_row = chunk * 16
        end_row   = start_row + 16
        
        # Lấy 16 hàng output của chunk này
        # Shape con: (16, 128)
        sub_matrix = w_matrix[start_row:end_row, :]
        
        # Duyệt theo CỘT INPUT (0 -> 127)
        for col in range(D_INNER):
            # Cột dọc 16 phần tử (16 output channels tương ứng với 1 input channel này)
            col_data = sub_matrix[:, col]
            
            # Ghi vào danh sách (16 dòng này sẽ được Hardware đọc trong 1 cycle)
            for val in col_data:
                hex_lines.append(float_to_hex(val))

    # 3. Ghi ra file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(hex_lines))
        
    print(f"DONE! Saved to {OUTPUT_FILE}")
    print(f"Total lines: {len(hex_lines)}")
    # Expected: 64 * 128 = 8192 dòng
    print(f"Expected lines: {D_MODEL * D_INNER} (8192)")

if __name__ == "__main__":
    run()