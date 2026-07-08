import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
CHANNELS = 128
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- INPUT FILES (Các file Golden Token-First gốc) ---
# Sửa đường dẫn tới file X và Gate mà cậu dùng để check Phase 1, 2
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/"
FILE_X_IN    = "conv_y_golden_ptb.txt"  # File này có thể đã là Channel-First? Check kỹ!
FILE_GATE_IN = "linear2_golden.txt"     # File này chắc chắn Token-First (Output Linear)

# --- OUTPUT FILES (Cho TB Scan Phase 4) ---
FILE_X_OUT    = "scan_x_channel_first.txt"
FILE_GATE_OUT = "scan_gate_channel_first.txt"

def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def process_file(file_in, file_out, name):
    print(f"Processing {name}...")
    try:
        with open(file_in) as f:
            data_flat = [to_signed(l.strip()) for l in f if l.strip()]
        
        # 1. Xác định Shape gốc
        # Nếu file X là Conv Out (từ phần cứng), nó có thể đã là Channel-First?
        # Nếu file Gate là Linear Out, nó là Token-First (1000, 128).
        
        # Giả sử file Input là TOKEN-FIRST (Chuẩn PyTorch)
        # Shape: (1000 Tokens, 128 Channels)
        matrix_tk = np.array(data_flat).reshape(SEQ_LEN, CHANNELS)
        
        # 2. Transpose sang CHANNEL-FIRST
        # Shape: (128 Channels, 1000 Tokens)
        matrix_ch = matrix_tk.T 
        
        # 3. Flatten & Save
        hex_lines = []
        for ch in range(CHANNELS):
            for t in range(SEQ_LEN):
                hex_lines.append(to_hex(matrix_ch[ch, t]))
                
        with open(file_out, 'w') as f:
            f.write('\n'.join(hex_lines))
            
        print(f"-> Saved {file_out} (Channel-First order)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Lưu ý: Nếu conv_y_golden_ptb.txt CỦA CẬU đã là Channel-First (do script cũ tạo ra) 
    # thì không cần transpose X nữa. Nhưng Gate thì chắc chắn cần.
    # Cứ chạy thử Transpose cả 2, nếu sai thì đảo lại.
    
    # Check Gate (Bắt buộc)
    process_file(FILE_GATE_IN, FILE_GATE_OUT, "Gate")
    
    # Check X (Cẩn thận: Kiểm tra xem conv_y_golden_ptb là T-F hay C-F)
    # Nếu Conv Hardware Output -> Nó là Channel First.
    # Nếu Conv Golden Python -> Nó thường là Channel First (Batch, Ch, L).
    # -> Nếu X đã là Channel First thì script này sẽ làm sai (Transpose thành Token First).
    # -> HÃY THỬ CHẠY GATE TRƯỚC.
    
    # process_file(FILE_X_IN, FILE_X_OUT, "X")