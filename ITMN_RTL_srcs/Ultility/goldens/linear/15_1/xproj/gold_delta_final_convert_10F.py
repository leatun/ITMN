import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
# Input Shape: (Channel=128, Time=1000)
CHANNELS = 128
SEQ_LEN = 1000

FRAC_BITS = 10
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE ---
# File gốc từ C++/Python (như trong ảnh cậu gửi)
FILE_IN = "D:/DoAn1/Ultility/goldens/cpp_golden_files/10_09_Mixer_delta_final.txt"

# File Output cho Testbench
FILE_OUT = "gold_delta_final_10F.txt"

# ==============================================================================
# HÀM XỬ LÝ
# ==============================================================================
def float_to_hex(val):
    val = int(round(val * SCALE))
    if val > MAX_INT: val = MAX_INT
    if val < MIN_INT: val = MIN_INT
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def run():
    print(f"Reading {FILE_IN}...")
    
    # 1. Đọc File vào Matrix
    # Giả sử file lưu dạng: 128 dòng (channel), mỗi dòng 1000 số (time)
    try:
        matrix = np.loadtxt(FILE_IN)
    except Exception as e:
        print(f"Error loading numpy: {e}")
        return

    print(f"Original Shape: {matrix.shape}")

    # 2. Kiểm tra và Transpose
    # Hardware cần: Loop Time -> Loop Channel (Token-First)
    # Nếu file gốc là (128, 1000) -> Cần Transpose thành (1000, 128)
    if matrix.shape == (CHANNELS, SEQ_LEN):
        print("Detected (Channel, Time). Transposing to (Time, Channel)...")
        matrix_hw = matrix.T 
    elif matrix.shape == (SEQ_LEN, CHANNELS):
        print("Detected (Time, Channel). No Transpose needed.")
        matrix_hw = matrix
    else:
        print(f"ERROR: Shape {matrix.shape} không khớp (128, 1000) hay (1000, 128)!")
        return

    print(f"Hardware Shape: {matrix_hw.shape} (Time, Channel)")

    # 3. Flatten và Convert sang Hex
    # Thứ tự mong muốn trong file text output:
    # Token 0: Ch0, Ch1 ... Ch15, Ch16 ... Ch127
    # Token 1: ...
    # Điều này khớp với logic (Token * 8 + Chunk) * 16 + k trong Testbench
    
    hex_lines = []
    
    for t in range(SEQ_LEN):
        for c in range(CHANNELS):
            val = matrix_hw[t, c]
            hex_lines.append(float_to_hex(val))
            
    # 4. Ghi File
    with open(FILE_OUT, 'w') as f:
        f.write('\n'.join(hex_lines))
        
    print(f"DONE! Saved to {FILE_OUT}")
    print(f"Total lines: {len(hex_lines)} (Should be 128000)")

if __name__ == "__main__":
    run()