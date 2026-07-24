# ============================================================================
# run_synth_impl_mamba2.tcl
#   Full OOC flow: synth → impl (place+route) → utilization + timing reports.
#   Target: KV260 (xck26-sfvc784-2LV-c), Mamba2 port of Mamba_Top.
#   Assumes project already loaded (open_project first if not).
#
# Usage from Vivado Tcl Console:
#   source run_synth_impl_mamba2.tcl
# ============================================================================

set proj_dir  "C:/Users/ADMIN/Downloads/DoAn1/DoAn1/ITMN_FASTMAMBA"
set src_lut   "C:/Users/ADMIN/Downloads/DoAn1/DoAn1/ITMN_re/golden_all"
set rpt_dir   "${proj_dir}/reports_mamba2"

# ---------- 0. Prep: copy LUT ROMs to project root so $readmemh finds them ----
file mkdir "${proj_dir}/golden_all"
foreach f {silu_lut.txt softplus_lut.txt exp_lut.txt rsqrt_q97.txt} {
    if {[file exists "${src_lut}/$f"]} {
        file copy -force "${src_lut}/$f" "${proj_dir}/golden_all/$f"
        puts "Copied $f"
    } else {
        puts "WARN: ${src_lut}/$f not found"
    }
}

# Also add golden_all/ to sim + synth include search
set_property include_dirs [list \
    "${proj_dir}/common" \
    "${proj_dir}/golden_all" \
] [get_filesets sources_1]

# ---------- 1. Force OOC mode on synth_1 + impl_1 ----------------------------
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-mode out_of_context} -objects [get_runs synth_1]

# ---------- 2. Report directory ---------------------------------------------
file mkdir $rpt_dir

# ---------- 3. Reset + run synth_1 ------------------------------------------
puts "=========================================="
puts "  Starting synth_1 (OOC mode)"
puts "=========================================="
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: synth_1 failed. Check runme.log"
    return
}

open_run synth_1 -name synth_1

# ---------- 4. Post-synth reports -------------------------------------------
puts "  Writing post-synth reports..."
report_utilization    -file "${rpt_dir}/post_synth_util.rpt"
report_timing_summary -file "${rpt_dir}/post_synth_timing.rpt"
report_timing -sort_by group -max_paths 10 -path_type full_clock \
              -file "${rpt_dir}/post_synth_top10_paths.rpt"
close_design

# ---------- 5. Reset + run impl_1 in OOC ------------------------------------
puts "=========================================="
puts "  Starting impl_1 (place + route, OOC)"
puts "=========================================="
reset_run impl_1
launch_runs impl_1 -to_step route_design -jobs 8
wait_on_run impl_1

set impl_progress [get_property PROGRESS [get_runs impl_1]]
if {$impl_progress != "100%"} {
    puts "WARN: impl_1 stopped at $impl_progress. Trying to open partial..."
}

open_run impl_1 -name impl_1

# ---------- 6. Post-route (or best-available) reports -----------------------
puts "  Writing post-route reports..."
report_utilization    -file "${rpt_dir}/post_route_util.rpt"
report_timing_summary -file "${rpt_dir}/post_route_timing.rpt"
report_timing -sort_by group -max_paths 20 -path_type full_clock \
              -file "${rpt_dir}/post_route_top20_paths.rpt"
report_power          -file "${rpt_dir}/post_route_power.rpt"
report_clock_utilization -file "${rpt_dir}/post_route_clock.rpt"

# ---------- 7. Extract fmax + key numbers to console ------------------------
set slack [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
if {$slack == ""} { set slack 0.0 }
set target_period 5.000
set achieved_period [expr {$target_period - $slack}]
set fmax_mhz [expr {1000.0 / $achieved_period}]

puts "=========================================="
puts "  Mamba2 SYNTH+IMPL SUMMARY"
puts "=========================================="
puts "  Target period : $target_period ns (200 MHz)"
puts "  Worst slack   : $slack ns"
puts "  Achieved      : $achieved_period ns"
puts "  fmax          : [format %.2f $fmax_mhz] MHz"
puts ""
puts "  Reports in: $rpt_dir"
puts "  Files:"
puts "    post_synth_util.rpt / post_synth_timing.rpt"
puts "    post_route_util.rpt / post_route_timing.rpt"
puts "    post_route_top20_paths.rpt / post_route_power.rpt"
puts "=========================================="
