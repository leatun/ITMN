import numpy as np
import os

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
SCALE = 1 << 12
MAX_INT = 32767
MIN_INT = -32768

# FILE INPUT (File output của Conv+SiLU từ C++ - Shape 128x1000)
INPUT_FILE = "D:/DoAn1/Ultility/goldens/cpp_golden_files/09_08_Mixer_x_activated.txt"
OUTPUT_FILE = "x_prime_chunked_verified.txt" # Đổi tên file output để tránh nhầm

def to_hex(val):
    val = int(np.round(val * SCALE))
    val = np.clip(val, MIN_INT, MAX_INT)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def run():
    print("Generating X_Prime Input for Phase 3 TB (VERIFIED)...")
    
    # 1. Load Data
    raw = np.loadtxt(INPUT_FILE)
    
    # Đảm bảo data là [Channel, Time] (128, 1000)
    if raw.shape == (SEQ_LEN, D_INNER): 
        data = raw.T 
    else:
        data = raw   
        
    print(f"Data Shape: {data.shape} (Rows=Channels, Cols=Tokens)")
    
    # --- CHECK PREVIEW ---
    print("\n--- PREVIEW DATA SOURCE ---")
    print(f"Ch0, T0: {data[0,0]}")
    print(f"Ch1, T0: {data[1,0]}")
    print(f"Ch0, T1: {data[0,1]}")
    print("---------------------------")

    hex_lines = []
    debug_lines = [] # Để in ra màn hình kiểm tra
    
    num_chunks = D_INNER // 16 # 8
    
    # Cấu trúc RAM: 8 Chunks.
    # Chunk 0: Chứa Channel 0-15.
    # Dòng 0 của Chunk 0: Phải chứa Ch0..15 tại Time 0.
    
    for chunk in range(num_chunks):
        start_ch = chunk * 16
        
        for t in range(SEQ_LEN):
            # Lấy 16 channel tại thời điểm t
            for i in range(16):
                ch = start_ch + i
                val = data[ch, t]
                hex_str = to_hex(val)
                hex_lines.append(hex_str)
                
                # Lưu thông tin debug cho 32 dòng đầu tiên
                if len(hex_lines) <= 32:
                    debug_lines.append(f"Line {len(hex_lines)-1}: Val={hex_str} (Src: Ch{ch}, T{t})")

    # 3. Ghi file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(hex_lines))
        
    print(f"\nDONE! Saved to {OUTPUT_FILE}")
    print(f"Total lines: {len(hex_lines)}")
    
    print("\n--- FILE STRUCTURE CHECK (MUST MATCH CONTROLLER) ---")
    for l in debug_lines:
        print(l)
    print("----------------------------------------------------")
    print("HÃY KIỂM TRA: Dòng 0 phải là Ch0 T0, Dòng 1 phải là Ch1 T0...")
    print("NẾU Dòng 1 là Ch0 T1 -> SAI (Đây là format cũ).")

if __name__ == "__main__":
    run()