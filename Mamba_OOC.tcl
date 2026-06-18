# ============================================================================
# Mamba_OOC.tcl — Out-of-context synth + impl for D1 standalone Mamba block.
#
# Run from Vivado tcl shell (or batch):
#   vivado -mode batch -source Mamba_OOC.tcl
#
# Produces standalone Mamba resource & timing reports for the paper's
# fair-compare table:
#   ./reports/mamba/utilization_placed.rpt
#   ./reports/mamba/timing_summary_routed.rpt
#   ./reports/mamba/route_status.rpt
#
# Mechanism: read ITM_CONTROLLER_v2.v + wrapper Mamba_Top.v with verilog define
# MAMBA_ONLY. The controller's `ifdef MAMBA_ONLY` arms strip the P1/BR/FIN/
# CASCADE state code; opt_design DCEs unused datapath. Memory_System,
# PE_Array, Const_Storage instantiate identically (DMA paths still drive all
# RAMs) — only FSM/datapath logic shrinks.
# ============================================================================

set RTL_DIR  ITMN_RTL_srcs/sources_1/new
set CONSTR   ITMN_RTL_srcs/constrs_1/new/itmn.xdc
set OUT_DIR  reports/mamba
set TOP      Mamba_Top
set PART     xck26-sfvc784-2LV-c

file mkdir $OUT_DIR

# ---- Read sources (NO original ITM_CONTROLLER.v — we use v2 fork) ----
read_verilog -sv \
    $RTL_DIR/_parameter.v \
    $RTL_DIR/_block_params.v \
    $RTL_DIR/BRAM_256b.v \
    $RTL_DIR/Memory_System.v \
    $RTL_DIR/Silu_LUT.v \
    $RTL_DIR/Softplus_LUT.v \
    $RTL_DIR/Exp_LUT.v \
    $RTL_DIR/RSqrt_ROM.v \
    $RTL_DIR/Const_Storage.v \
    $RTL_DIR/PE_BLOCK.v \
    $RTL_DIR/Unified_PE.v \
    $RTL_DIR/RMSNorm_Mul.v \
    $RTL_DIR/ITM_CONTROLLER_v2.v \
    $RTL_DIR/Mamba_Top.v

read_xdc $CONSTR

# ---- Synth with MAMBA_ONLY define, out-of-context (no I/O buffers) ----
synth_design -top $TOP -part $PART \
    -mode out_of_context \
    -verilog_define MAMBA_ONLY

# ---- Place + Route ----
opt_design
place_design
route_design

# ---- Reports ----
report_utilization -file $OUT_DIR/utilization_placed.rpt
report_timing_summary -file $OUT_DIR/timing_summary_routed.rpt
report_route_status -file $OUT_DIR/route_status.rpt
report_clock_utilization -file $OUT_DIR/clock_utilization.rpt

# Save post-route checkpoint for UI inspection later:
#   open_checkpoint reports/mamba/mamba_top_routed.dcp
write_checkpoint -force $OUT_DIR/mamba_top_routed.dcp

puts "----------------------------------------------------------------"
puts "  Mamba_Top OOC done."
puts "  Reports   : $OUT_DIR/"
puts "  Checkpoint: $OUT_DIR/mamba_top_routed.dcp"
puts "----------------------------------------------------------------"
