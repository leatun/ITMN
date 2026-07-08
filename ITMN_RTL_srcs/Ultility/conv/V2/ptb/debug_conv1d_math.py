import numpy as np

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128      # Số kênh của Conv1D
D_LINEAR_OUT = 256 # Số kênh trong file 08_X (128 Conv + 128 Gate)
KERNEL_SIZE = 4

# --- ĐƯỜNG DẪN FILE (SỬA LẠI ĐƯỜNG DẪN TRÊN MÁY CẬU) ---
FILE_INPUT_X  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/08_X_after_linear.txt"
FILE_GOLDEN_Y = "D:/DoAn1/Ultility/goldens/cpp_golden_files/09_08_Mixer_x_activated.txt"
FILE_WEIGHT   = "D:/DoAn1/Ultility/conv/V2/ptb/conv1d_weight.txt"
FILE_BIAS     = "D:/DoAn1/Ultility/conv/V2/ptb/conv1d_bias.txt"

# --- HÀM TOÁN HỌC ---
def silu_float(x):
    # SiLU = x / (1 + e^-x)
    try:
        return x * (1.0 / (1.0 + np.exp(-x)))
    except OverflowError:
        return 0.0 if x < 0 else x

def run_debug():
    print("=== DEBUG CONV1D MATH (PURE FLOAT) ===")

    # 1. LOAD INPUT (CẮT DỮ LIỆU)
    print(f"1. Loading Input from {FILE_INPUT_X}...")
    try:
        x_raw = np.loadtxt(FILE_INPUT_X)
        print(f"   Raw Shape: {x_raw.shape} (Expected 1000, 256)")
        
        # --- CẮT DỮ LIỆU Ở ĐÂY ---
        # Lấy 1000 dòng, 128 cột đầu tiên
        X = x_raw[:, :D_INNER] 
        print(f"   Sliced Input Shape: {X.shape} (For Conv1D)")
        
    except Exception as e:
        print(f"   [ERR] Load Input: {e}"); return

    # 2. LOAD GOLDEN OUTPUT
    print(f"2. Loading Golden Output from {FILE_GOLDEN_Y}...")
    try:
        y_raw = np.loadtxt(FILE_GOLDEN_Y)
        # Check shape: C++ có thể lưu (128, 1000) hoặc (1000, 128)
        if y_raw.shape == (D_INNER, SEQ_LEN):
            Y_gold = y_raw.T # Transpose về (1000, 128)
        else:
            Y_gold = y_raw.reshape(SEQ_LEN, D_INNER)
        print(f"   Golden Y Shape: {Y_gold.shape}")
    except Exception as e:
        print(f"   [ERR] Load Golden: {e}"); return

    # 3. LOAD WEIGHT (XỬ LÝ PARSE)
    print("3. Parsing Weights...")
    W = np.zeros((D_INNER, KERNEL_SIZE))
    try:
        with open(FILE_WEIGHT, 'r') as f:
            lines = f.readlines()
            # Parse từng dòng, bỏ qua index [x,x,x]
            idx_count = 0
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                val_str = parts[-1] # Lấy số cuối cùng
                val = float(val_str)
                
                # Map vào mảng (Assuming linear order: C0_K0..3, C1_K0..3)
                c = idx_count // KERNEL_SIZE
                k = idx_count % KERNEL_SIZE
                if c < D_INNER:
                    W[c, k] = val
                idx_count += 1
        print(f"   Weights Loaded. Shape: {W.shape}")
    except Exception as e:
        print(f"   [ERR] Load Weight: {e}"); return

    # 4. LOAD BIAS
    print("4. Parsing Bias...")
    B = np.zeros((D_INNER))
    try:
        # Bias có thể dính index hoặc không, dùng cách an toàn nhất
        with open(FILE_BIAS, 'r') as f:
            lines = f.readlines()
            idx_count = 0
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                val = float(parts[-1])
                if idx_count < D_INNER:
                    B[idx_count] = val
                idx_count += 1
        print(f"   Bias Loaded. Shape: {B.shape}")
    except Exception as e:
        print(f"   [ERR] Load Bias: {e}"); return

    # --- TÍNH TOÁN ---
    print("\n--- COMPUTING (Python Float64) ---")
    Y_calc = np.zeros_like(Y_gold)
    
    for c in range(D_INNER):
        w_c = W[c] # [w0, w1, w2, w3]
        b_c = B[c]
        x_c = X[:, c] # Vector 1000 phần tử
        
        # History buffer: [x_t, x_t-1, x_t-2, x_t-3]
        # Khởi tạo toàn 0
        history = [0.0] * KERNEL_SIZE
        
        for t in range(SEQ_LEN):
            # Cập nhật lịch sử
            history.insert(0, x_c[t])
            history.pop()
            
            # Tính Conv: Bias + Sum(History * Weight)
            # Tính Conv: Bias + Sum(History * Weight)
            val = b_c
            for k in range(KERNEL_SIZE):
                if t - k >= 0: # (Thực ra history đã handle việc này, nhưng cứ để k)
                    # CŨ (SAI): 
                    # val += history[k] * w_c[k]
                    
                    # MỚI (ĐẢO CHIỀU WEIGHT):
                    # history[0] là x hiện tại (t), nhân với w[3]
                    # history[3] là x cũ nhất (t-3), nhân với w[0]
                    val += history[k] * w_c[KERNEL_SIZE - 1 - k]
            
            # Activation SiLU
            Y_calc[t, c] = silu_float(val)

    # --- SO SÁNH ---
    print("\n--- COMPARISON RESULTS ---")
    diff = np.abs(Y_calc - Y_gold)
    max_diff = np.max(diff)
    
    print(f"Max Difference (Float vs Float): {max_diff:.6f}")
    
    # In ra vài mẫu để kiểm tra
    print(f"Sample T=0, C=0: Calc={Y_calc[0,0]:.4f}, Gold={Y_gold[0,0]:.4f}, Diff={diff[0,0]:.4f}")
    print(f"Sample T=10, C=0: Calc={Y_calc[10,0]:.4f}, Gold={Y_gold[10,0]:.4f}, Diff={diff[10,0]:.4f}")

    if max_diff < 1e-3:
        print("\n>>> SUCCESS: Logic toán học ĐÚNG! Sai số cực nhỏ.")
        print(">>> Bạn có thể yên tâm dùng logic này để sinh file Golden Hex.")
    else:
        print("\n>>> FAILURE: Sai số quá lớn! Có thể sai thứ tự Weight hoặc Padding.")
        print(">>> Gợi ý: Thử đảo ngược thứ tự Weight trong vòng lặp tính toán.")

if __name__ == "__main__":
    run_debug()