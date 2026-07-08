import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

# --- ĐƯỜNG DẪN FILE ---
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/"
DIR_CPP    = "D:/DoAn1/Ultility/goldens/cpp_golden_files/"

# 1. Hardware Golden (Do gen_scan_golden_full_v2.py tạo ra - Hex, Channel-First)
FILE_HW_SCAN = "gold_scan_final.txt"

# 2. PTB Golden (Do extract_single_sample.py tạo ra - Float, Shape 128x1000)
# Tên file trong script của cậu là "14_Mixer_y_gated.txt"
FILE_PTB_SCAN = DIR_CPP + "1014_14_Mixer_y_gated.txt"

# ==============================================================================
# HELPER
# ==============================================================================
def to_signed(hex_str):
    val = int(hex_str, 16)
    return val - 65536 if val & 0x8000 else val

def float_to_fixed(val):
    return int(round(val * SCALE))

# ==============================================================================
# MAIN CHECK
# ==============================================================================
def run():
    print("--- CROSS-CHECK: HARDWARE LOGIC vs PYTORCH REALITY ---")
    
    # 1. Load HW Scan (Channel-First from Hex)
    print(f"Loading HW Scan: {FILE_HW_SCAN}")
    with open(FILE_HW_SCAN) as f:
        hw_flat = [to_signed(l.strip()) for l in f if l.strip()]
    
    # Reshape về (Channel, Time) vì script gen lưu channel-first
    HW_Matrix = np.array(hw_flat).reshape(D_INNER, SEQ_LEN)
    print(f"HW Shape: {HW_Matrix.shape} (Channel, Time)")

    # 2. Load PTB Scan (Float Matrix)
    print(f"Loading PTB Scan: {FILE_PTB_SCAN}")
    try:
        # Load dạng matrix. Script PTB lưu shape [1, 128, 1000] -> squeeze -> [128, 1000]
        # Hoặc [1000, 128] tùy cách save_tensor.
        # Trong script extract: save_tensor(y_gated, "14_Mixer_y_gated")
        # y_gated shape trong code: [1, 128, 1000] (channel-first) -> save_tensor reshape -> 128 dòng, 1000 cột?
        # Check logic save_tensor: if ndim > 2: reshape(-1, shape[-1])
        # [1, 128, 1000] -> [128, 1000]. => Dòng = Channel, Cột = Time.
        
        PTB_Float = np.loadtxt(FILE_PTB_SCAN)
        if PTB_Float.shape != (D_INNER, SEQ_LEN):
            print(f"WARNING: PTB Shape {PTB_Float.shape} != (128, 1000). Transposing...")
            PTB_Float = PTB_Float.T
            
        # Convert PTB Float -> Fixed để so sánh công bằng
        PTB_Fixed = np.vectorize(float_to_fixed)(PTB_Float)
        print(f"PTB Shape: {PTB_Fixed.shape} (Channel, Time)")
        
    except Exception as e:
        print(f"Error loading PTB: {e}"); return

    # 3. Compare
    print("\n--- COMPARING (CHANNEL-WISE) ---")
    
    # HW: (128, 1000), PTB: (128, 1000) -> Khớp shape
    diff = np.abs(HW_Matrix - PTB_Fixed)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    print(f"MAX Diff: {max_diff}")
    print(f"AVG Diff: {avg_diff:.4f}")
    
    # Check sample channel 0
    print("\nSample Channel 0 (First 10 timesteps):")
    print(f"HW : {HW_Matrix[0, :10]}")
    print(f"PTB: {PTB_Fixed[0, :10]}")
    
    # Check sample channel 64
    print("\nSample Channel 64 (First 10 timesteps):")
    print(f"HW : {HW_Matrix[64, :10]}")
    print(f"PTB: {PTB_Fixed[64, :10]}")

    if max_diff > 1000:
        print("\n>>> KẾT LUẬN: LỆCH NẶNG! <<<")
        print("Có thể do:")
        print("1. Sai công thức SSM (Discrete A, B).")
        print("2. Sai thứ tự Input (Delta/B/C bị lệch kênh hoặc lệch thời gian).")
        print("3. Sai Gate (SiLU approximation).")
        print("4. Delta chưa qua Softplus? (Check xem gold_delta_final đã softplus chưa)")
    else:
        print("\n>>> KẾT LUẬN: CHẤP NHẬN ĐƯỢC (Fixed-point Error). <<<")

if __name__ == "__main__":
    run()