import numpy as np
import os
import glob
import math

# ===================================================================
# 1. CẤU HÌNH
# ===================================================================
TENSOR_DIR = "cpp_golden_files/"
TOTAL_BITS_W = 16
PERCENTILE_THRESHOLD = 99.99

# ===================================================================
# 2. HÀM TÍNH BIT PHẦN NGUYÊN
# ===================================================================
def get_required_integer_bits(max_abs_value: float) -> int:
    """
    Tính số bit phần nguyên (bao gồm sign bit)
    cần để biểu diễn |value| lớn nhất
    """
    if max_abs_value < 1.0:
        return 1  # chỉ cần sign bit
    value_bits = math.ceil(math.log2(max_abs_value))
    return int(value_bits) + 1  # +1 cho sign bit

# ===================================================================
# 3. MAIN
# ===================================================================
if __name__ == "__main__":

    print("\n--- Bat dau khao sat dai gia tri ---\n")

    # Header bảng
    header = (
        "{:<50} | {:>12} | {:>12} | {:>18} | {:>12} | {:>12} | {:<18}"
        .format(
            "Tensor name",
            "Min",
            "Max",
            "Clip@{:.2f}%".format(PERCENTILE_THRESHOLD),
            "I bits",
            "F bits",
            "Proposed type"
        )
    )
    print(header)
    print("-" * len(header))

    # Biến tổng hợp
    max_integer_bits_needed = 0
    max_fractional_bits_needed = 0
    tensor_requiring_max_I = ""
    tensor_requiring_max_F = ""

    # Lấy danh sách file
    file_paths = sorted(glob.glob(os.path.join(TENSOR_DIR, "*.txt")))
    if not file_paths:
        print("❌ Khong tim thay file .txt trong thu muc:", TENSOR_DIR)
        exit(1)

    # ===================================================================
    # 4. DUYỆT TỪNG TENSOR
    # ===================================================================
    for filepath in file_paths:
        filename = os.path.basename(filepath)

        try:
            data = np.loadtxt(filepath)
            if data.size == 0:
                print(f"{filename:<50} | FILE RONG")
                continue

            min_val = np.min(data)
            max_val = np.max(data)

            # Percentile clipping
            abs_data = np.abs(data).flatten()
            sorted_abs = np.sort(abs_data)

            clip_index = int(len(sorted_abs) * PERCENTILE_THRESHOLD / 100.0) - 1
            clip_index = max(0, min(clip_index, len(sorted_abs) - 1))

            clip_threshold = sorted_abs[clip_index]

            # Fixed-point analysis
            integer_bits_I = get_required_integer_bits(clip_threshold)
            fractional_bits_F = TOTAL_BITS_W - integer_bits_I
            proposed_type = f"ap_fixed<{TOTAL_BITS_W}, {integer_bits_I}>"

            # Update global max
            if integer_bits_I > max_integer_bits_needed:
                max_integer_bits_needed = integer_bits_I
                tensor_requiring_max_I = filename

            if fractional_bits_F > max_fractional_bits_needed:
                max_fractional_bits_needed = fractional_bits_F
                tensor_requiring_max_F = filename

            # Print row
            print(
                "{:<50} | {:>12.5f} | {:>12.5f} | {:>18.5f} | {:>12} | {:>12} | {:<18}"
                .format(
                    filename,
                    min_val,
                    max_val,
                    clip_threshold,
                    integer_bits_I,
                    fractional_bits_F,
                    proposed_type
                )
            )

        except Exception as e:
            print(f"{filename:<50} | ❌ Loi: {e}")

    # ===================================================================
    # 5. TÓM TẮT KẾT QUẢ
    # ===================================================================
    print("\n" + "=" * 60)
    print(" KET QUA KHAO SAT TONG THE")
    print("=" * 60)

    print("\n[PHAN NGUYEN - INTEGER PART]")
    print(f"So bit phan nguyen lon nhat can thiet (I_max): {max_integer_bits_needed}")
    print(f" -> Tensor yeu cau: '{tensor_requiring_max_I}'")

    safe_integer_bits = max_integer_bits_needed
    safe_fractional_bits = TOTAL_BITS_W - safe_integer_bits

    print("\n[DE XUAT DINH DANG AN TOAN - ONE SIZE FITS ALL]")
    print("De dam bao KHONG bi overflow o bat ky layer nao:")
    print(f" - Tong bit W = {TOTAL_BITS_W}")
    print(f" - Integer bits = {safe_integer_bits}")
    print(f" - Fractional bits = {safe_fractional_bits}")
    print(f"\n==> DE XUAT: ap_fixed<{TOTAL_BITS_W}, {safe_integer_bits}>")

    print("\n[CANH BAO VE DO CHINH XAC]")
    print(f"So bit fractional LON NHAT can thiet: {max_fractional_bits_needed}")
    print(f" -> Yeu cau boi tensor: '{tensor_requiring_max_F}'")
    print(
        "Neu fractional bits an toan ({}) < yeu cau ({}), "
        "se co MAT MAT do chinh xac."
        .format(safe_fractional_bits, max_fractional_bits_needed)
    )

    print("\n--- Khao sat hoan tat ---\n")
