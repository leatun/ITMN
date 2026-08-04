# ============================================================================
# run_synth_only.tcl
#   OOC synth only - no impl. Dumps hierarchical util for BRAM diagnosis.
#   Target: KV260 (xck26-sfvc784-2LV-c), Mamba_Top Phase 6.
#
# Usage from Vivado Tcl Console (with project open):
#   source C:/Users/ADMIN/Downloads/DoAn1/DoAn1/ITMN_FASTMAMBA/run_synth_only.tcl
# ============================================================================

set proj_dir  "C:/Users/ADMIN/Downloads/DoAn1/DoAn1/ITMN_FASTMAMBA"
set rpt_dir   "${proj_dir}/reports_mamba2"

file mkdir $rpt_dir

# Force OOC mode on synth_1
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-mode out_of_context} -objects [get_runs synth_1]

puts "==========================================="
puts "  Starting synth_1 (OOC mode, NO impl)"
puts "==========================================="
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synth_1 failed. Check runme.log"
    return
}

open_run synth_1 -name synth_1

puts "  Writing post-synth reports..."
report_utilization -file "${rpt_dir}/post_synth_util.rpt"
report_utilization -hierarchical -file "${rpt_dir}/post_synth_util_hier.rpt"
report_utilization -hierarchical -hierarchical_depth 3 \
    -file "${rpt_dir}/post_synth_util_hier_d3.rpt"
report_timing_summary -file "${rpt_dir}/post_synth_timing.rpt"

puts ""
puts "==========================================="
puts "  Quick BRAM/URAM/DSP summary"
puts "==========================================="
set util [report_utilization -return_string]
foreach line [split $util "\n"] {
    if {[regexp {RAMB36|RAMB18|URAM|DSP48|CLB LUT|CLB Reg} $line]} {
        puts $line
    }
}
puts "==========================================="
puts ""
puts "  Full reports in: $rpt_dir"
