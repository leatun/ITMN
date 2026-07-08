import math

# --- CAU HINH Q3.12 ---
TOTAL_BITS = 16
FRAC_BITS = 12
SCALE = 1 << FRAC_BITS

def int_to_float(int_val):
    # two's complement 16-bit -> signed int
    if int_val >= (1 << (TOTAL_BITS - 1)):
        int_val -= (1 << TOTAL_BITS)
    return int_val / SCALE

def float_to_int(float_val):
    val = int(round(float_val * SCALE))
    max_val = (1 << (TOTAL_BITS - 1)) - 1
    min_val = -(1 << (TOTAL_BITS - 1))

    if val > max_val: val = max_val
    if val < min_val: val = min_val

    # signed -> two's complement unsigned
    if val < 0:
        val = (1 << TOTAL_BITS) + val
    return val

def sigmoid(x):
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

print(f"Generating silu_rom.mem for Q3.{FRAC_BITS}...")

with open("silu_rom.mem", "w") as f:
    for i in range(1 << TOTAL_BITS):
        real_input = int_to_float(i)

        # SiLU = x * sigmoid(x)
        real_output = real_input * sigmoid(real_input)

        fixed_output = float_to_int(real_output)
        f.write(f"{fixed_output:04x}\n")

print("Done! Copy 'silu_rom.mem' to your Vivado project.")
