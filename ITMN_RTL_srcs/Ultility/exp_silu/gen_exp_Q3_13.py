import math

# --- CẤU HÌNH Q3.12 ---
TOTAL_BITS = 16
FRAC_BITS = 12 #
SCALE = 1 << FRAC_BITS 

# Hàm chuyển từ số nguyên (máy hiểu) sang số thực
def int_to_float(int_val):
    # Xử lý số âm (Two's complement) 16-bit
    if int_val >= (1 << (TOTAL_BITS - 1)):
        int_val -= (1 << TOTAL_BITS)
    return int_val / SCALE

# Hàm chuyển từ số thực sang số nguyên (máy hiểu)
def float_to_int(float_val):
    # Nhân với Scale và làm tròn
    val = int(round(float_val * SCALE))
    # Giới hạn (Clamping) để không bị tràn 16-bit
    max_val = (1 << (TOTAL_BITS - 1)) - 1
    min_val = -(1 << (TOTAL_BITS - 1))
    if val > max_val: val = max_val
    if val < min_val: val = min_val
    # Xử lý số âm sang dạng Hex (Two's complement)
    if val < 0: val = (1 << TOTAL_BITS) + val
    return val

# --- TẠO FILE ROM ---
print(f"Generating exp_rom.mem for Q2.{FRAC_BITS}...")

with open("exp_rom.mem", "w") as f:
    # Chạy từ 0000 đến FFFF (tất cả các bit pattern có thể có của 16-bit)
    # i ở đây chính là địa chỉ của ROM, cũng chính là giá trị thô của input
    for i in range(1 << TOTAL_BITS):
        
        # 1. Hiểu i là số thực bao nhiêu?
        real_input = int_to_float(i)
        
        # 2. Tính e mũ x
        # Lưu ý: nếu input dương quá lớn, exp sẽ tràn. 
        # Nhưng trong Mamba, delta * A thường là số âm, nên an toàn.
        try:
            real_output = math.exp(real_input)
        except OverflowError:
            real_output = 9999.0 # Gán giá trị max nếu tràn

        # 3. Chuyển kết quả về lại Q2.13
        fixed_output = float_to_int(real_output)
        
        # 4. Ghi vào file (chỉ ghi giá trị hex)
        f.write(f"{fixed_output:04x}\n")

print("Done! Copy 'exp_rom.mem' to your Vivado project.")