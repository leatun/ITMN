import numpy as np
import math
import re

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
SEQ_LEN = 1000
D_INNER = 128
D_STATE = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS
MAX_INT = 32767
MIN_INT = -32768

# --- FILES ---
DIR_GOLDEN = "D:/DoAn1/Ultility/goldens/" # Sửa đường dẫn nếu cần
DIR_VEC    = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/"

# Input từ RAM Dump (Do TB sinh ra)
FILE_X_DUMP     = "dump_x_ram.txt"
FILE_GATE_DUMP  = "dump_gate_ram.txt"
FILE_DELTA_DUMP = "dump_delta_ram.txt"
FILE_B_DUMP     = "dump_b_ram.txt"
FILE_C_DUMP     = "dump_c_ram.txt"

# Tham số tĩnh (Vẫn lấy từ file gốc)
FILE_A_LOG = DIR_VEC + "A_log.txt"
FILE_D     = DIR_VEC + "D.txt"

# File Golden chuẩn để so sánh
FILE_GOLD_FINAL = "gold_scan_final.txt"

# ==============================================================================
# HELPER
# ==============================================================================
def to_signed(val):
    return val - (1 << 16) if val & 0x8000 else val

def parse_line_256(hex_line):
    """Tách dòng hex 256-bit thành 16 giá trị int 16-bit"""
    # Hex string có thể dài ngắn khác nhau, pad 0 cho đủ 64 ký tự (256 bit)
    hex_line = hex_line.strip().zfill(64)
    vals = []
    # Phần cứng lưu: [Channel 15]...[Channel 0] (MSB...LSB)
    # Python cắt từ trái qua phải là MSB -> LSB.
    # Nên channel 0 nằm ở cuối chuỗi.
    for i in range(16):
        # Cắt 4 ký tự (16 bit) từ cuối lên
        start = 60 - (i * 4)
        chunk = hex_line[start : start+4]
        vals.append(to_signed(int(chunk, 16)))
    return vals # Trả về [Ch0, Ch1, ..., Ch15]

def float_to_fixed(val):
    v = int(round(val * SCALE))
    return max(min(v, MAX_INT), MIN_INT)

def fixed_mul(a, b):
    return (a * b) >> FRAC_BITS

def silu_fixed_approx(x_fixed):
    x_real = x_fixed / SCALE
    sigmoid = 1.0 / (1.0 + math.exp(-x_real))
    return int(x_fixed * sigmoid)

# ==============================================================================
# RECONSTRUCT DATA FROM RAM
# ==============================================================================
def load_ram_dump():
    print("--- Reconstructing Data from RAM Dumps ---")
    
    # 1. Load X & Gate (Group-First Layout)
    # RAM: [Group 0 (1000 lines)] [Group 1] ...
    # Mỗi line chứa 16 Channels.
    X_matrix = np.zeros((D_INNER, SEQ_LEN), dtype=int)
    Gate_matrix = np.zeros((D_INNER, SEQ_LEN), dtype=int)
    
    def load_group_first(fname, matrix):
        with open(fname) as f:
            lines = f.readlines()
        
        line_idx = 0
        for group in range(8): # 8 Groups
            for t in range(SEQ_LEN): # 1000 Tokens
                vals = parse_line_256(lines[line_idx])
                line_idx += 1
                
                # Gán vào matrix
                # Group 0: Channel 0-15
                base_ch = group * 16
                for k in range(16):
                    matrix[base_ch + k, t] = vals[k]
                    
    load_group_first(FILE_X_DUMP, X_matrix)
    load_group_first(FILE_GATE_DUMP, Gate_matrix)
    print("-> X & Gate Loaded (Group-First Logic)")

    # 2. Load Delta (Token-First Strided Layout)
    # RAM: [Token 0 (8 chunks)] [Token 1] ...
    Delta_matrix = np.zeros((SEQ_LEN, D_INNER), dtype=int)
    
    with open(FILE_DELTA_DUMP) as f:
        lines = f.readlines()
        
    line_idx = 0
    for t in range(SEQ_LEN):
        for group in range(8):
            vals = parse_line_256(lines[line_idx])
            line_idx += 1
            
            base_ch = group * 16
            for k in range(16):
                Delta_matrix[t, base_ch + k] = vals[k]
    print("-> Delta Loaded (Token-First Logic)")

    # 3. Load B & C (Token-First Linear)
    # RAM: [Token 0] [Token 1] ... (16 values per line)
    B_matrix = np.zeros((SEQ_LEN, D_STATE), dtype=int)
    C_matrix = np.zeros((SEQ_LEN, D_STATE), dtype=int)
    
    def load_simple(fname, matrix):
        with open(fname) as f: lines = f.readlines()
        for t in range(SEQ_LEN):
            vals = parse_line_256(lines[t])
            for k in range(16): matrix[t, k] = vals[k]
            
    load_simple(FILE_B_DUMP, B_matrix)
    load_simple(FILE_C_DUMP, C_matrix)
    print("-> B & C Loaded")
    
    return X_matrix, Gate_matrix, Delta_matrix, B_matrix, C_matrix

def load_static_params():
    # Load A & D (như cũ)
    A_fixed = np.zeros((D_INNER, D_STATE), dtype=int)
    D_fixed = np.zeros(D_INNER, dtype=int)
    
    try:
        with open(FILE_A_LOG) as f:
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) >= 3:
                    r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
                    val = -math.exp(v)
                    val = max(-8.0, min(7.999, val)) # Clamping like HW
                    A_fixed[r, c] = float_to_fixed(val)
    except: pass

    try:
        with open(FILE_D) as f:
            idx = 0
            for line in f:
                parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(parts) == 1: 
                    D_fixed[idx] = float_to_fixed(float(parts[0])); idx+=1
                elif len(parts)>=2: 
                    D_fixed[int(parts[0])] = float_to_fixed(float(parts[1]))
    except: pass
    
    return A_fixed, D_fixed

# ==============================================================================
# RUN SIMULATION & COMPARE
# ==============================================================================
def run():
    # 1. Get Data from RAM Dumps
    X, Gate, Delta, B, C = load_ram_dump()
    A, D = load_static_params()
    
    print("\n--- Running Scan Logic on RAM Dump Data ---")
    
    Y_calc = []
    
    for d in range(D_INNER):
        h = np.zeros(D_STATE, dtype=int)
        A_row = A[d]
        D_val = D[d]
        
        for t in range(SEQ_LEN):
            u = X[d, t]
            delta = Delta[t, d]
            gate = Gate[d, t] # Gate HW lưu giống X
            B_t = B[t]
            C_t = C[t]
            
            # --- SSM ---
            disc_A = np.zeros(D_STATE, dtype=int)
            dBx = np.zeros(D_STATE, dtype=int)
            
            for n in range(D_STATE):
                # A bar
                dA_fixed = fixed_mul(delta, A_row[n])
                dA_float = dA_fixed / SCALE
                disc_A[n] = float_to_fixed(math.exp(dA_float))
                
                # B bar x
                dB = fixed_mul(delta, B_t[n])
                dBx[n] = fixed_mul(dB, u)
            
            # Update H
            h_new = np.zeros(D_STATE, dtype=int)
            for n in range(D_STATE):
                val = fixed_mul(disc_A[n], h[n]) + dBx[n]
                h_new[n] = max(min(val, MAX_INT), MIN_INT)
            h = h_new
            
            # Output
            y_scan = 0
            for n in range(D_STATE):
                y_scan += fixed_mul(C_t[n], h[n])
                
            du = fixed_mul(D_val, u)
            y_with_D = y_scan + du
            
            g_act = silu_fixed_approx(gate)
            y_out = fixed_mul(y_with_D, g_act)
            
            # Clamp output
            y_out = max(min(y_out, MAX_INT), MIN_INT)
            Y_calc.append(y_out)

    print("--- Loading Golden Reference ---")
    with open(FILE_GOLD_FINAL) as f:
        Y_gold = [int(l.strip(), 16) for l in f if l.strip()]
        # Convert hex to signed
        Y_gold = [val - 65536 if val & 0x8000 else val for val in Y_gold]

    print("--- Comparing ---")
    diffs = []
    for i in range(len(Y_calc)):
        d = abs(Y_calc[i] - Y_gold[i])
        diffs.append(d)
        if d > 100 and i < 10:
            print(f"Mismatch at {i}: Calc={Y_calc[i]}, Gold={Y_gold[i]}, Diff={d}")
            
    print(f"MAX Diff: {max(diffs)}")
    print(f"AVG Diff: {np.mean(diffs):.2f}")
    
    if max(diffs) == 0:
        print("\n>>> RESULT: EXACT MATCH! RAM Data is CORRECT. <<<")
        print("Hardware Phase 4 is reading WRONG or Computing WRONG.")
    else:
        print("\n>>> RESULT: MISMATCH! RAM Data is INCORRECT. <<<")
        print("Hardware Phase 1/2/3 produced wrong data.")

if __name__ == "__main__":
    run()