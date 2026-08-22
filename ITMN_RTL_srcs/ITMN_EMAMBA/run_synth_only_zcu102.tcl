# ============================================================================
# run_synth_only_zcu102.tcl
#   OOC synth-only on ZCU102 project. Writes utilization + timing reports.
#   Assumes project already created via setup_project_zcu102.tcl.
#
# Usage: source run_synth_only_zcu102.tcl
# ============================================================================

set proj_dir [file normalize [file dirname [info script]]]
set rpt_dir  "${proj_dir}/reports_zcu102"
file mkdir $rpt_dir

set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-mode out_of_context} -objects [get_runs synth_1]

puts "==========================================="
puts "  Launching synth_1 (ZCU102, OOC, no impl)"
puts "==========================================="
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synth_1 failed. Check runme.log in .runs/synth_1/"
    return
}

open_run synth_1 -name synth_1

puts "  Writing post-synth reports..."
report_utilization        -file "${rpt_dir}/post_synth_util.rpt"
report_utilization -hier  -file "${rpt_dir}/post_synth_util_hier.rpt"
report_timing_summary     -file "${rpt_dir}/post_synth_timing.rpt"
report_power              -file "${rpt_dir}/post_synth_power.rpt"

puts ""
puts "==========================================="
puts "  Quick util summary (ZCU102)"
puts "==========================================="
set util [report_utilization -return_string]
foreach line [split $util "\n"] {
    if {[regexp {RAMB36|RAMB18|DSP48|LUT as Logic|LUT as Memory|CLB Reg} $line]} {
        puts $line
    }
}
puts "==========================================="
puts "  Full reports in: $rpt_dir"
