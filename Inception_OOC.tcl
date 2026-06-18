# ============================================================================
# Inception_OOC.tcl — Out-of-context synth + impl for D1 standalone Inception.
#
# Run:
#   vivado -mode batch -source Inception_OOC.tcl
#
# Produces standalone Inception resource & timing reports:
#   ./reports/inception/utilization_placed.rpt
#   ./reports/inception/timing_summary_routed.rpt
#
# Mechanism: ITM_CONTROLLER_v2.v compiled with INCEPTION_ONLY define strips the
# M*/NORM Mamba state arms; S_BR_NEXT after last branch jumps directly to
# S_FIN_READ. FIN reads A_MAMBA_OUT (never written) → BRAM init 0 → relu(0)=0
# → final = relu(bn(inc)) = standalone Inception block output.
# ============================================================================

set RTL_DIR  ITMN_RTL_srcs/sources_1/new
set CONSTR   ITMN_RTL_srcs/constrs_1/new/itmn.xdc
set OUT_DIR  reports/inception
set TOP      Inception_Top
set PART     xck26-sfvc784-2LV-c

file mkdir $OUT_DIR

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
    $RTL_DIR/Inception_Top.v

read_xdc $CONSTR

synth_design -top $TOP -part $PART \
    -mode out_of_context \
    -verilog_define INCEPTION_ONLY

opt_design
place_design
route_design

report_utilization -file $OUT_DIR/utilization_placed.rpt
report_timing_summary -file $OUT_DIR/timing_summary_routed.rpt
report_route_status -file $OUT_DIR/route_status.rpt
report_clock_utilization -file $OUT_DIR/clock_utilization.rpt

# Save post-route checkpoint for UI inspection later:
#   open_checkpoint reports/inception/inception_top_routed.dcp
write_checkpoint -force $OUT_DIR/inception_top_routed.dcp

puts "----------------------------------------------------------------"
puts "  Inception_Top OOC done."
puts "  Reports   : $OUT_DIR/"
puts "  Checkpoint: $OUT_DIR/inception_top_routed.dcp"
puts "----------------------------------------------------------------"
