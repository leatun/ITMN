# ============================================================================
# itmn.xdc - Timing for Mamba_Top v3 (sources_v3) after MUL2 refactor.
#
# Target : KV260 SOM (xck26-sfvc784-2LV-c, UltraScale+ -1LV)
# Target : 100 MHz (10.0 ns period)
#
# History:
#   - 200 MHz initial attempt failed with WNS = -8.19 ns.
#     Critical path (29 logic levels, 13.19 ns):
#       URAM(ram_main) -> DSP48 mult -> CARRY8x6 -> LUT6x10 -> m6_dA_reg
#   - MUL2 refactor (S_M6_DAB_MUL2 + EXP_WAIT + DAB_LATCH) fed Exp_LUT from
#     a registered cl_out_vec instead of combinational cl_out_next_vec.
#     Expected new path: PE.reg CLK-Q -> Exp_LUT -> m6_dA_reg (~3-4 ns).
#   - 100 MHz leaves ~6 ns headroom above that.
#
# If 100 MHz fails, likely next candidates (also use cl_out_next_vec):
#   - S_MAC write path (Σacc -> m_wr_data -> ram_main)
#   - S_M6_T1/T2/DU_LATCH (h·dA_reg, u·dB_reg, D·u)
# ============================================================================

# ----------------------------------------------------------------------------
# Primary clock - 100 MHz
# ----------------------------------------------------------------------------
create_clock -name clk -period 10.000 -waveform {0.000 5.000} [get_ports clk]

set_clock_uncertainty -setup 0.150 [get_clocks clk]
set_clock_uncertainty -hold  0.050 [get_clocks clk]

# ----------------------------------------------------------------------------
# Async reset - assert async, deassert sync
# ----------------------------------------------------------------------------
set_false_path -from [get_ports rst]
