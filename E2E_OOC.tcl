# ============================================================================
# E2E_OOC.tcl — Out-of-context synth + impl for D2 full end-to-end controller
# (ITM_Top_v3: encoder + 5 ITM blocks + GAP + FC head).
#
# Run from Vivado tcl shell (or batch):
#   vivado -mode batch -source E2E_OOC.tcl
#
# Produces full-flow resource & timing reports for the paper's
# end-to-end vs kernel-only comparison:
#   ./reports/v3/utilization_placed.rpt
#   ./reports/v3/timing_summary_routed.rpt
#   ./reports/v3/route_status.rpt
#   ./reports/v3/clock_utilization.rpt
#   ./reports/v3/itm_top_v3_routed.dcp
#
# Mechanism: identical to D1 OOC scripts but without any define (full v3
# build).  Same Memory_System, PE_Array, Const_Storage primitives — Const_Storage
# ram_const is the v3 version (depth 128 instead of 64, to hold encoder+FC bias).
#
# IMPORTANT: this build uses ITM_CONTROLLER_v3.v, NOT v1/v2.  v3 has 8-bit
# state register and the new S_ENC_*/S_GAP_*/S_FC_* state arms.
# ============================================================================

set RTL_DIR  ITMN_RTL_srcs/sources_1/new
set CONSTR   ITMN_RTL_srcs/constrs_1/new/itmn.xdc
set OUT_DIR  reports/v3
set TOP      ITM_Top_v3
set PART     xck26-sfvc784-2LV-c

file mkdir $OUT_DIR

# ---- Read sources (NO v1/v2 controller — v3 is the only one for full E2E) ----
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
    $RTL_DIR/ITM_CONTROLLER_v3.v

read_xdc $CONSTR

# ---- Synth (full E2E, no define) ----
synth_design -top $TOP -part $PART -mode out_of_context

# ---- Place + Route ----
opt_design
place_design
route_design

# ---- Reports ----
report_utilization        -file $OUT_DIR/utilization_placed.rpt
report_timing_summary     -file $OUT_DIR/timing_summary_routed.rpt
report_route_status       -file $OUT_DIR/route_status.rpt
report_clock_utilization  -file $OUT_DIR/clock_utilization.rpt

# Hierarchical breakdown so D2_Compare can attribute LUT/REG/DSP to
# encoder/GAP/FC sub-paths vs the existing ITM body.
report_utilization        -hierarchical -hierarchical_depth 3 \
                          -file $OUT_DIR/utilization_hier.rpt

# Save post-route checkpoint for later UI inspection / re-analysis.
write_checkpoint -force $OUT_DIR/itm_top_v3_routed.dcp

puts "----------------------------------------------------------------"
puts "  ITM_Top_v3 E2E OOC done."
puts "  Reports   : $OUT_DIR/"
puts "  Checkpoint: $OUT_DIR/itm_top_v3_routed.dcp"
puts "----------------------------------------------------------------"
