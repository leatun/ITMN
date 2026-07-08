import numpy as np

# --- CẤU HÌNH ---
D_MODEL = 64     # 64 Cột (Input dim)
D_INNER = 128    # 128 Hàng (Output dim)
DATA_WIDTH = 16

# --- ĐƯỜNG DẪN FILE CŨ (Row-Major) ---
FILE_W_OLD = "D:/DoAn1/Ultility/goldens/linear_weight_proj2.mem"

# --- ĐƯỜNG DẪN FILE MỚI (Column-Major / Folded) ---
FILE_W_NEW = "lin_real_w.txt" # Nạp cái này vào BRAM

def run():
    print("=== REORDERING WEIGHT FOR CONTROLLER ===")
    
    # 1. Đọc file cũ (Hex) -> Chuyển về list
    try:
        with open(FILE_W_OLD, 'r') as f:
            w_hex_list = [line.strip() for line in f if line.strip()]
    except Exception as e: print(f"Err: {e}"); return

    # 2. Reshape về Ma trận (128 Hàng, 64 Cột)
    # File cũ lưu theo hàng: Hàng 0 (64 số), Hàng 1 (64 số)...
    w_matrix = np.array(w_hex_list).reshape(D_INNER, D_MODEL)
    print(f"Loaded Weight Matrix: {w_matrix.shape}")
    
    # 3. Sắp xếp lại (Reorder)
    # Controller chạy theo 8 Chunks.
    # Chunk 0: Xử lý Hàng 0..15.
    # Trong Chunk 0: Chạy 64 nhịp (Cột 0..63).
    # Tại nhịp k: Cần lấy Cột k của các Hàng 0..15.
    
    new_hex_list = []
    
    for chunk in range(8): # 0..7
        start_row = chunk * 16
        end_row   = start_row + 16
        
        # Lấy ma trận con (16, 64)
        sub_matrix = w_matrix[start_row:end_row, :]
        
        # Duyệt theo Cột (0..63)
        for col in range(D_MODEL):
            # Lấy 16 phần tử của cột 'col'
            # Đây chính là W_row_vals cho nhịp 'col'
            # [W[0][col], W[1][col], ... W[15][col]]
            
            # Lưu vào list (Flatten)
            # Lưu ý: Testbench/Controller đọc 256-bit (16 số) từ 1 địa chỉ.
            # Ta ghi 16 dòng hex liên tiếp, hay ghi 1 dòng dài?
            # Script testbench cậu dùng $readmemh, nó đọc từng dòng hex 16-bit.
            # Nên ta cứ ghi 16 dòng liên tiếp. BRAM sẽ tự gom 16 dòng thành 1 từ nhớ.
            
            col_data = sub_matrix[:, col] # 16 phần tử
            for val in col_data:
                new_hex_list.append(val)

    # 4. Ghi file mới
    with open(FILE_W_NEW, 'w') as f:
        f.write('\n'.join(new_hex_list))
        
    print(f"DONE! Saved reordered weights to {FILE_W_NEW}")
    print(f"Total lines: {len(new_hex_list)} (Should be 8192)")

if __name__ == "__main__":
    run()