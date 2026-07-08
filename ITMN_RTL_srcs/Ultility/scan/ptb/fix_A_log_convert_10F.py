import numpy as np
import math

# --- CẤU HÌNH ---
D_INNER = 128
D_STATE = 16
DATA_WIDTH = 16
FRAC_BITS = 10
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- ĐƯỜNG DẪN FILE ---
# File A_log gốc bị dính index
FILE_A_LOG_INPUT = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/A_log.txt"
# File Output cho Verilog
FILE_HEX_OUT     = "scan_real_A_10F.txt"

# --- HÀM HỖ TRỢ ---
def to_hex(val):
    val = int(val)
    if val < 0: val = (1 << 16) + val
    return f"{val & 0xFFFF:04x}"

def float_to_fixed(val):
    val = np.round(val * SCALE).astype(int)
    val = np.clip(val, MIN_INT, MAX_INT)
    return val

def run():
    print(f"Reading & Fixing A_log from: {FILE_A_LOG_INPUT}...")
    
    vals_list = []
    
    try:
        with open(FILE_A_LOG_INPUT, 'r') as f:
            lines = f.readlines()
            
        print(f"  -> Found {len(lines)} lines.")
        
        # 1. Parse Text (Bỏ index [x,y])
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # Lấy phần tử cuối cùng là giá trị số
                val_str = parts[-1]
                vals_list.append(float(val_str))
                
        # Kiểm tra số lượng phần tử
        expected = D_INNER * D_STATE # 128 * 16 = 2048
        if len(vals_list) != expected:
            print(f"  [WARNING] Số lượng phần tử: {len(vals_list)} (Kỳ vọng: {expected})")
            
        # 2. Convert sang Numpy
        # Shape gốc là (128, 16)
        A_log_arr = np.array(vals_list).reshape(D_INNER, D_STATE)
        
        # 3. Tính toán A thực tế
        # Công thức C++: A = -exp(A_log)
        print("  -> Computing A = -exp(A_log)...")
        A_real = -np.exp(A_log_arr)
        
        # 4. Convert sang Fixed Point
        print("  -> Converting to Q3.12 Hex...")
        A_fixed = float_to_fixed(A_real)
        
        # 5. Ghi file Hex
        with open(FILE_HEX_OUT, "w") as f:
            for d in range(D_INNER):
                for n in range(D_STATE):
                    # Ghi từng dòng một (Flatten)
                    f.write(to_hex(A_fixed[d, n]) + "\n")
                    
        print(f"DONE! Saved to: {FILE_HEX_OUT}")
        print(f"Sample Hex (A[0][0]): {to_hex(A_fixed[0,0])} (Float: {A_real[0,0]:.4f})")

    except FileNotFoundError:
        print("ERROR: Không tìm thấy file input.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run()