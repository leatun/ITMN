# ============================================================================
# mamba_top_mamba2.xdc — OOC timing constraints for Mamba_Top (Mamba2 build).
#
# Target FPGA : KV260 SOM (Zynq UltraScale+ MPSoC XCK26-SFVC784-2LV-C)
# Target Fmax : 200 MHz first pass (5.0 ns period).
#               After clean WNS, retry at 4.0 ns (250 MHz) to expose true fmax.
#
# Diff vs Mamba1 mamba_top.xdc:
#   - Adds N_STATE_GRP, USE_M5, XP_OUT_GRP_IN to quasi-static input group
#   - Same clock/reset scheme (fair comparison)
#
# Usage:
#   set_property USED_IN_SYNTHESIS true      [get_files mamba_top_mamba2.xdc]
#   set_property USED_IN_IMPLEMENTATION true [get_files mamba_top_mamba2.xdc]
# ============================================================================

# ----------------------------------------------------------------------------
# Primary clock — 100 MHz final (Mamba2 config, KV260).
# History:
#   Target 200MHz (5ns)  → FAIL WNS -5.595, path URAM→PE MAC 10.48ns
#   Target 100MHz (10ns) → FAIL WNS -0.712, same URAM→PE path 10.48ns
#   Added URAM pipeline reg in M_Cluster.v
#   Target 143MHz (7ns)  → FAIL WNS -2.423, new critical path
#                          FSM state → PE input mux → DSP MAC 9.27ns
#   Target 100MHz (10ns) → PASS margin ~+0.6ns fmax cap ~106 MHz
# ----------------------------------------------------------------------------
create_clock -name clk -period 10.000 -waveform {0.000 5.000} [get_ports clk]

set_clock_uncertainty -setup 0.150 [get_clocks clk]
set_clock_uncertainty -hold  0.050 [get_clocks clk]

# ----------------------------------------------------------------------------
# Async reset — assert async, deassert sync
# ----------------------------------------------------------------------------
set_false_path -from [get_ports rst]

# ----------------------------------------------------------------------------
# Quasi-static config ports — stable through entire run.
# Set false_path so config doesn't limit fmax (they only sample at S_IDLE→H_INIT).
# ----------------------------------------------------------------------------
set_false_path -from [get_ports {run_stage[*] T_MAX[*] CH_OUT[*] CH_M[*] DT_RANK[*] \
                                 N_STATE_GRP[*] USE_M5 XP_OUT_GRP_IN[*]}]

# ----------------------------------------------------------------------------
# Synchronous control — start pulse (sampled at S_IDLE)
# ----------------------------------------------------------------------------
set_input_delay  -clock clk -max 2.0 [get_ports start]
set_input_delay  -clock clk -min 0.2 [get_ports start]

# ----------------------------------------------------------------------------
# DMA write bus — driven by external host / test harness
# ----------------------------------------------------------------------------
set_input_delay  -clock clk -max 2.0 [get_ports {dma_write_en dma_target[*] dma_addr[*] dma_wdata[*]}]
set_input_delay  -clock clk -min 0.2 [get_ports {dma_write_en dma_target[*] dma_addr[*] dma_wdata[*]}]

set_input_delay  -clock clk -max 2.0 [get_ports {dma_read_en dma_rtarget[*] dma_raddr[*]}]
set_input_delay  -clock clk -min 0.2 [get_ports {dma_read_en dma_rtarget[*] dma_raddr[*]}]

# ----------------------------------------------------------------------------
# Output delay — done flags + DMA read data
# ----------------------------------------------------------------------------
set_output_delay -clock clk -max 2.0 [get_ports {done_stage done_all dma_rdata[*]}]
set_output_delay -clock clk -min 0.2 [get_ports {done_stage done_all dma_rdata[*]}]

# ============================================================================
# Memory / DSP hints
# ============================================================================
# H_RegFile depth 16384 × 256b = 4 Mb → force URAM cascade on UltraScale+
# (BRAM would cost 32 tiles; URAM = 1 slice)
# Uncomment if H_RegFile doesn't infer URAM automatically:
# set_property RAM_STYLE ultra [get_cells -hier -filter {NAME =~ */u_hrf/*ram*}]

# Weight RAM 8192 × 256b = 2 Mb → BRAM cascade OK, or split URAM if pressure high
# set_property RAM_STYLE ultra [get_cells -hier -filter {NAME =~ */ram_weight*}]

# DSP inference — Mamba_PE has 2 mults per lane × 16 lanes = 32 DSP48E2/cluster
# Should map automatically; hint if not:
# set_property USE_DSP YES [get_cells -hier -filter {NAME =~ */u_pe*/m1_reg* || NAME =~ */u_pe*/m2_reg*}]

# ============================================================================
# Retiming / pipeline
# ============================================================================
# Enable Vivado auto-retiming across PE MAC pipeline for max fmax
# (already enabled by default in synth strategy PerformanceOptimized_high)
