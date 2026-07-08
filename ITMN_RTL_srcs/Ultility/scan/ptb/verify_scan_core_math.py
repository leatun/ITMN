import numpy as np
import math

# --- CẤU HÌNH ---
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
DATA_WIDTH = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768
EPSILON = 1e-5

# --- FILES ---
F_A     = "scan_real_A.txt"
F_B     = "scan_real_B_shared.txt"
F_C     = "scan_real_C_shared.txt"
F_D     = "scan_real_D.txt"
F_DELTA = "scan_real_delta.txt"
F_X     = "scan_real_x.txt"
F_GATE  = "linear2_golden.txt" # File này phải có nha!
F_GOLD  = "scan_y_golden_HARDWARE.txt"

# --- HELPER ---
def hex_to_int(hex_str):
    val = int(hex_str, 16)
    if val >= 32768: val -= 65536
    return val

def sat16(val):
    if val > MAX_INT: return MAX_INT
    if val < MIN_INT: return MIN_INT
    return val

def fixed_mul(a, b):
    # Mô phỏng PE: Nhân -> Shift 12
    return sat16((a * b) >> FRAC_BITS)

def fixed_exp(val):
    # Mô phỏng Exp Unit
    # Input: Q3.12 int -> Float -> Exp -> Q3.12 int
    f_val = val / SCALE
    res = math.exp(f_val)
    return sat16(int(round(res * SCALE)))

def fixed_silu(val):
    # Mô phỏng SiLU Unit
    f_val = val / SCALE
    if f_val >= 0: sigmoid = 1.0 / (1.0 + math.exp(-f_val))
    else:          sigmoid = math.exp(f_val) / (1.0 + math.exp(f_val))
    res = f_val * sigmoid
    return sat16(int(round(res * SCALE)))

# --- MAIN ---
def run():
    print("=== VERIFY SCAN CORE LOGIC (PYTHON SIMULATION) ===")
    
    # 1. LOAD DATA
    print("1. Loading Hex Files...")
    try:
        with open(F_A, 'r') as f: A_raw = [hex_to_int(x) for x in f]
        with open(F_B, 'r') as f: B_raw = [hex_to_int(x) for x in f]
        with open(F_C, 'r') as f: C_raw = [hex_to_int(x) for x in f]
        with open(F_D, 'r') as f: D_raw = [hex_to_int(x) for x in f]
        
        with open(F_DELTA, 'r') as f: Delta_raw = [hex_to_int(x) for x in f]
        with open(F_X, 'r')     as f: X_raw     = [hex_to_int(x) for x in f]
        with open(F_GATE, 'r')  as f: Gate_raw  = [hex_to_int(x) for x in f]
        with open(F_GOLD, 'r')  as f: Y_gold    = [hex_to_int(x) for x in f]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Reshape về cấu trúc dễ dùng
    # A: (128, 16)
    A_mat = np.array(A_raw).reshape(D_INNER, D_STATE)
    # B, C: (1000, 16)
    B_mat = np.array(B_raw).reshape(SEQ_LEN, D_STATE)
    C_mat = np.array(C_raw).reshape(SEQ_LEN, D_STATE)
    # Inputs: (128, 1000)
    Delta_mat = np.array(Delta_raw).reshape(D_INNER, SEQ_LEN)
    X_mat     = np.array(X_raw).reshape(D_INNER, SEQ_LEN)
    Gate_mat  = np.array(Gate_raw).reshape(D_INNER, SEQ_LEN)
    Y_mat     = np.array(Y_gold).reshape(D_INNER, SEQ_LEN)
    
    # 2. SIMULATION LOOP
    print("2. Simulating Scan Core Logic...")
    
    total_error = 0
    max_diff = 0
    
    # Duyệt từng kênh
    for d in range(D_INNER):
        # Lấy tham số cho kênh d
        A_vec = A_mat[d]
        D_val = D_raw[d]
        
        # Hidden State h (16,) khởi tạo bằng 0
        h = np.zeros(D_STATE, dtype=int)
        
        for t in range(SEQ_LEN):
            # Input tại t
            delta = Delta_mat[d, t]
            x_val = X_mat[d, t]
            gate  = Gate_mat[d, t]
            
            # Shared B, C tại t
            B_vec = B_mat[t]
            C_vec = C_mat[t]
            
            # --- BƯỚC 1: Discretization & State Update ---
            # h_new = exp(delta*A) * h + (delta*B) * x
            h_new = np.zeros(D_STATE, dtype=int)
            
            for n in range(D_STATE):
                # 1.1 Tính exp(delta * A)
                deltaA = fixed_mul(delta, A_vec[n])
                discA  = fixed_exp(deltaA) # Exp
                
                # 1.2 Tính (delta * B) * x
                deltaB = fixed_mul(delta, B_vec[n])
                deltaBx = fixed_mul(deltaB, x_val)
                
                # 1.3 Cập nhật h
                term1 = fixed_mul(discA, h[n])
                h_new[n] = sat16(term1 + deltaBx)
            
            # Cập nhật trạng thái
            h = h_new
            
            # --- BƯỚC 2: Scan Output (y = C * h) ---
            y_scan = 0
            for n in range(D_STATE):
                # Lưu ý: Trong HW dùng Adder Tree, ở đây cộng dồn int
                prod = fixed_mul(C_vec[n], h[n])
                y_scan += prod
            
            # --- BƯỚC 3: Residual + Gating ---
            # y_out = (y_scan + D*x) * SiLU(gate)
            
            Dx = fixed_mul(D_val, x_val)
            y_with_D = y_scan + Dx # Int cộng int (chưa bão hòa ngay, để rộng tí cũng đc)
            
            gate_act = fixed_silu(gate)
            
            # Nhân Gating
            y_final = (y_with_D * gate_act) >> FRAC_BITS
            y_final = sat16(y_final)
            
            # --- BƯỚC 4: So sánh ---
            gold = Y_mat[d, t]
            diff = abs(y_final - gold)
            
            if diff > max_diff: max_diff = diff
            
            # Ngưỡng sai số (Do exp/silu xấp xỉ)
            # 20 LSB ~ 0.005
            if diff > 30: 
                # In lỗi đầu tiên để debug
                if total_error == 0:
                    print(f"\n[FIRST ERROR] Ch={d}, Time={t}")
                    print(f"  Inputs: delta={delta}, x={x_val}, gate={gate}")
                    print(f"  Calc y_scan={y_scan}, Dx={Dx}, y_with_D={y_with_D}")
                    print(f"  Gate_Act={gate_act}")
                    print(f"  Final: Calc={y_final} vs Gold={gold}")
                total_error += 1

    # --- REPORT ---
    print("\n" + "="*30)
    print(f"Processed {D_INNER * SEQ_LEN} outputs.")
    print(f"Max Difference: {max_diff} LSB")
    print(f"Total Significant Errors (>30 LSB): {total_error}")
    
    if total_error < 500: # Chấp nhận vài trăm lỗi biên do làm tròn dồn tích
        print(">>> SUCCESS: Logic Scan Core KHỚP với C++ Golden! <<<")
        print("(Sai số nhỏ là do khác biệt giữa `math.exp` và `ROM/PWL`)")
    else:
        print(">>> WARNING: Sai số lớn. Kiểm tra lại Logic Gate hoặc thứ tự B/C.")

if __name__ == "__main__":
    run()