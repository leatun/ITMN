import numpy as np

# --- CẤU HÌNH ---
DATA_WIDTH = 16
FRAC_BITS = 12
MAX_VAL = (2**(DATA_WIDTH-1) - 1)
MIN_VAL = -(2**(DATA_WIDTH-1))
SCALE = 2**FRAC_BITS

INPUT_FILE  = "D:/DoAn1/Ultility/goldens/cpp_golden_files/06_06_MambaBlock_input.txt"
WEIGHT_FILE = "D:/DoAn1/Ultility/goldens/golden_vectors_txt/rms_norm_weight.txt" 
OUTPUT_FILE = "D:/DoAn1/Ultility/goldens/cpp_golden_files/07_07_MambaBlock_after_norm.txt"

def float_to_hex(f_val):
    # Scale & Saturate
    val = int(round(f_val * SCALE))
    if val > MAX_VAL: val = MAX_VAL
    if val < MIN_VAL: val = MIN_VAL
    
    # Two's complement for Hex
    if val < 0: val = (1 << DATA_WIDTH) + val
    return f"{val:04x}"

def process_file(input_path, output_filename, is_weight=False):
    print(f"Processing {input_path}...")
    hex_lines = []
    
    try:
        with open(input_path, 'r') as f:
            data = f.read().split() # Đọc toàn bộ số, tách theo khoảng trắng/newline
            
        for val_str in data:
            try:
                f_val = float(val_str)
                hex_lines.append(float_to_hex(f_val))
            except ValueError:
                continue # Bỏ qua nếu không phải số
                
        # Ghi ra file Hex cho Verilog
        with open(output_filename, 'w') as f:
            f.write("\n".join(hex_lines))
            
        print(f" -> Saved to {output_filename} ({len(hex_lines)} values)")
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_path}")

if __name__ == "__main__":
    # 1. Convert Input
    process_file(INPUT_FILE, "rms_ptb_input.txt")
    
    # 2. Convert Weight
    process_file(WEIGHT_FILE, "rms_ptb_weight.txt", is_weight=True)
    
    # 3. Convert Golden Output (Từ C++)
    process_file(OUTPUT_FILE, "rms_ptb_golden.txt")
    
    print("\nDONE! Copy 3 files .txt above to your Vivado Simulation folder.")