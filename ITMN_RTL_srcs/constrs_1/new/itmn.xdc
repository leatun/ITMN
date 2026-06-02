

# ---- Primary clock: 100 MHz (10 ns period) ----
create_clock -period 10.000 -name clk [get_ports clk]

# ---- Async reset: bá»? qua timing checks ----
set_false_path -from [get_ports rst]

