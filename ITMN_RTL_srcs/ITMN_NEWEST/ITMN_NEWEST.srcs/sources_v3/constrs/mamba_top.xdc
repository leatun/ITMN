# ============================================================================
# mamba_top.xdc — OOC (out-of-context) timing constraints for Mamba_Top.
#
# Target FPGA : KV260 SOM (Zynq UltraScale+ MPSoC XCK26-SFVC784-2LV-C)
# Target Fmax : 200 MHz baseline (5.0 ns period). UltraScale+ -1LV can typically
#               push 250 MHz on this kind of pipeline; bump period to 4.0 ns
#               after first clean run.
#
# Usage in Vivado:
#   set_property USED_IN_SYNTHESIS true  [get_files mamba_top.xdc]
#   set_property USED_IN_IMPLEMENTATION true [get_files mamba_top.xdc]
#
# OOC mode notes:
#   - No pin LOC / IOSTANDARD here — Mamba_Top is a sub-module, not chip top.
#   - clk is treated as a primary clock arriving at the Mamba_Top boundary.
#   - dma_* and run_stage/T_MAX/CH_OUT/CH_M/DT_RANK are quasi-static host writes;
#     timed but with relaxed input delay (host clock domain).
#   - rst is async assert / sync deassert at the controller; covered by set_false_path.
# ============================================================================

# ----------------------------------------------------------------------------
# Primary clock — 200 MHz target
# ----------------------------------------------------------------------------
create_clock -name clk -period 5.000 -waveform {0.000 2.500} [get_ports clk]

# Uncertainty budget — Vivado default is fine for OOC; explicit value below
# matches what Vivado adds for unknown-source clocks on UltraScale+.
set_clock_uncertainty -setup 0.150 [get_clocks clk]
set_clock_uncertainty -hold  0.050 [get_clocks clk]

# ----------------------------------------------------------------------------
# Async reset — assert async, deassert sync (host writes rst as level signal)
# ----------------------------------------------------------------------------
set_false_path -from [get_ports rst]

# ----------------------------------------------------------------------------
# DMA / control inputs — driven by external host (PS or test harness).
# In integrated SoC build, these come from AXI-Lite slave clocked by clk;
# during OOC synth we treat them as max ~40 % of period (2.0 ns at 200 MHz).
# ----------------------------------------------------------------------------
set_input_delay  -clock clk -max 2.0 [get_ports {start run_stage[*] T_MAX[*] CH_OUT[*] CH_M[*] DT_RANK[*]}]
set_input_delay  -clock clk -min 0.2 [get_ports {start run_stage[*] T_MAX[*] CH_OUT[*] CH_M[*] DT_RANK[*]}]

set_input_delay  -clock clk -max 2.0 [get_ports {dma_write_en dma_target[*] dma_addr[*] dma_wdata[*]}]
set_input_delay  -clock clk -min 0.2 [get_ports {dma_write_en dma_target[*] dma_addr[*] dma_wdata[*]}]

set_input_delay  -clock clk -max 2.0 [get_ports {dma_read_en dma_rtarget[*] dma_raddr[*]}]
set_input_delay  -clock clk -min 0.2 [get_ports {dma_read_en dma_rtarget[*] dma_raddr[*]}]

# ----------------------------------------------------------------------------
# Output delay — done flags + DMA read data go back to host.
# ----------------------------------------------------------------------------
set_output_delay -clock clk -max 2.0 [get_ports {done_stage done_all dma_rdata[*]}]
set_output_delay -clock clk -min 0.2 [get_ports {done_stage done_all dma_rdata[*]}]

# ============================================================================
# RAM style hints — let Vivado pick optimal mapping for KV260 (URAM available
# on UltraScale+). H_RegFile and Const_Storage already carry inline ram_style
# attributes ("block"); Memory_System main banks let the tool choose URAM vs
# BRAM cascade based on depth.
# ============================================================================

# ============================================================================
# DSP slice mapping — Mamba_PE has 2 multipliers per lane × 16 lanes = 32 DSP.
# UltraScale+ DSP48E2 should map automatically; if Vivado infers LUT mult
# instead, add: set_property USE_DSP YES [get_cells -hier -filter {NAME =~ */m1_reg* || NAME =~ */m2_reg*}]
# ============================================================================
