# ============================================================================
# D1_Compare.tcl — Open Mamba + Inception checkpoints, dump paper-ready
# artifacts to reports/d1_compare/.
#
# Run from Vivado tcl shell (Tcl Console or batch):
#   source D1_Compare.tcl
#
# Produces:
#   reports/d1_compare/
#     mamba_util_hier.rpt             hierarchical utilization (table)
#     mamba_critical_path.rpt         worst-case timing path
#     mamba_critical_path.pdf         schematic of that path  (paper figure)
#     inception_util_hier.rpt
#     inception_critical_path.rpt
#     inception_critical_path.pdf
#     d1_summary.txt                  side-by-side comparison table
#     d1_summary.csv                  same table, Excel/LaTeX-ready
# ============================================================================

set OUT_DIR reports/d1_compare
file mkdir $OUT_DIR

set MAMBA_DCP     reports/mamba/mamba_top_routed.dcp
set INCEPTION_DCP reports/inception/inception_top_routed.dcp

# Result containers
array set R {}

# ---------------------------------------------------------------------------
# Helper: parse a single number out of `report_utilization` text
# ---------------------------------------------------------------------------
proc grab_int {report regex} {
    if {[regexp -line $regex $report -> m]} { return $m }
    return 0
}

# ---------------------------------------------------------------------------
# Helper: per-design dump (hierarchical util + critical path text+schematic)
# ---------------------------------------------------------------------------
proc dump_design {label dcp_path out_dir} {
    global R
    puts "============================================================"
    puts "  Loading $label checkpoint: $dcp_path"
    puts "============================================================"
    open_checkpoint $dcp_path

    # 1. Hierarchical utilization (the table reviewers want)
    set util_rpt $out_dir/${label}_util_hier.rpt
    report_utilization -hierarchical -hierarchical_depth 3 -file $util_rpt
    puts "  -> $util_rpt"

    # 2. Capture top-level numbers for summary CSV
    set u_flat [report_utilization -return_string]
    set R(${label},LUT)   [grab_int $u_flat {\|\s*CLB LUTs\s+\|\s*(\d+)}]
    set R(${label},REG)   [grab_int $u_flat {\|\s*CLB Registers\s+\|\s*(\d+)}]
    set R(${label},CARRY) [grab_int $u_flat {\|\s*CARRY8\s+\|\s*(\d+)}]
    set R(${label},BRAM)  [grab_int $u_flat {\|\s*Block RAM Tile\s+\|\s*(\d+)}]
    set R(${label},URAM)  [grab_int $u_flat {\|\s*URAM\s+\|\s*(\d+)}]
    set R(${label},DSP)   [grab_int $u_flat {\|\s*DSPs\s+\|\s*(\d+)}]

    # 3. Critical path — text report
    set path_rpt $out_dir/${label}_critical_path.rpt
    report_timing -max_paths 1 -nworst 1 -delay_type max -path_type full \
        -input_pins -file $path_rpt
    puts "  -> $path_rpt"

    # 4. Capture WNS from timing summary
    set t_flat [report_timing_summary -return_string]
    if {[regexp {(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)} $t_flat -> wns tns whs ths]} {
        set R(${label},WNS) $wns
        set R(${label},WHS) $whs
    } else {
        set R(${label},WNS) "n/a"
        set R(${label},WHS) "n/a"
    }

    # 5. Critical-path schematic → PDF (paper figure)
    # Pull the single worst path's cells/nets and schematic-export them.
    set worst_path [get_timing_paths -max_paths 1 -nworst 1 -delay_type max]
    if {[llength $worst_path] > 0} {
        set path_cells [get_cells -of_objects $worst_path]
        set path_nets  [get_nets  -of_objects $worst_path]
        set path_objs  [concat $path_cells $path_nets]
        set sch_pdf    $out_dir/${label}_critical_path.pdf
        if {[catch {
            # write_schematic exports the currently-selected subset as PDF.
            select_objects $path_objs
            write_schematic -force -format pdf -file $sch_pdf
            unselect_objects
        } emsg]} {
            puts "  (!) write_schematic failed: $emsg"
            puts "      Open the checkpoint in GUI to capture schematic manually."
        } else {
            puts "  -> $sch_pdf"
        }
    }

    close_design
}

# ---------------------------------------------------------------------------
# Run both
# ---------------------------------------------------------------------------
dump_design mamba     $MAMBA_DCP     $OUT_DIR
dump_design inception $INCEPTION_DCP $OUT_DIR

# ---------------------------------------------------------------------------
# Side-by-side summary (text + CSV)
# ---------------------------------------------------------------------------
# Fixed reference: full ITM_Top result from c0d2d72 lock-in commit.
set R(full,LUT)   10491
set R(full,REG)   4504
set R(full,CARRY) 0   ;# not previously captured
set R(full,BRAM)  118
set R(full,URAM)  40
set R(full,DSP)   59
set R(full,WNS)   0.646
set R(full,WHS)   0.089

proc safe_get {key} { global R; if {[info exists R($key)]} { return $R($key) } else { return 0 } }

set txt $OUT_DIR/d1_summary.txt
set fh [open $txt w]
puts $fh "============================================================"
puts $fh "  D1 Separation — Standalone vs Integrated Resource Compare"
puts $fh "  Target: KV260 / xck26-sfvc784-2LV-c, OOC synth, clk=10 ns"
puts $fh "============================================================"
puts $fh ""
puts $fh "Resource     |    Mamba |  Incept. |     Full |  Sum-Full (shared)"
puts $fh "-------------+----------+----------+----------+-------------------"
foreach {key fmt} {LUT %d REG %d CARRY %d BRAM %d URAM %d DSP %d WNS %.3f WHS %.3f} {
    set m [safe_get mamba,$key]
    set i [safe_get inception,$key]
    set f [safe_get full,$key]
    set sum_minus_full ""
    if {$fmt eq "%d"} {
        set sum [expr {$m + $i}]
        set diff [expr {$sum - $f}]
        set sum_minus_full [format "%d (sum=%d)" $diff $sum]
    } else {
        set sum_minus_full "—"
    }
    puts $fh [format "%-12s | %8s | %8s | %8s | %s" \
        $key [format $fmt $m] [format $fmt $i] [format $fmt $f] $sum_minus_full]
}
puts $fh ""
puts $fh "Sharing analysis (logic-only, memory is fully shared by design):"
set lut_save [expr {$R(mamba,LUT) + $R(inception,LUT) - $R(full,LUT)}]
set lut_pct  [expr {round(100.0 * $lut_save / ($R(mamba,LUT) + $R(inception,LUT)))}]
set dsp_save [expr {$R(mamba,DSP) + $R(inception,DSP) - $R(full,DSP)}]
set dsp_pct  [expr {round(100.0 * $dsp_save / ($R(mamba,DSP) + $R(inception,DSP)))}]
set reg_save [expr {$R(mamba,REG) + $R(inception,REG) - $R(full,REG)}]
set reg_pct  [expr {round(100.0 * $reg_save / ($R(mamba,REG) + $R(inception,REG)))}]
puts $fh "  LUT shared : $lut_save ([format %d $lut_pct]% saved by fusion)"
puts $fh "  REG shared : $reg_save ([format %d $reg_pct]% saved by fusion)"
puts $fh "  DSP shared : $dsp_save ([format %d $dsp_pct]% saved by fusion)"
puts $fh "  Memory (BRAM/URAM): 100% shared (single Memory_System instance)"
close $fh
puts "  -> $txt"

# CSV — for Excel / LaTeX import
set csv $OUT_DIR/d1_summary.csv
set fh [open $csv w]
puts $fh "Resource,Mamba_Top,Inception_Top,Full_ITM,Mamba+Inception,Saved_by_fusion,Saved_pct"
foreach key {LUT REG CARRY BRAM URAM DSP} {
    set m [safe_get mamba,$key]; set i [safe_get inception,$key]; set f [safe_get full,$key]
    set sum [expr {$m + $i}]; set save [expr {$sum - $f}]
    set pct [expr {$sum > 0 ? round(100.0 * $save / $sum) : 0}]
    puts $fh "$key,$m,$i,$f,$sum,$save,$pct%"
}
puts $fh "WNS_ns,[safe_get mamba,WNS],[safe_get inception,WNS],[safe_get full,WNS],—,—,—"
puts $fh "WHS_ns,[safe_get mamba,WHS],[safe_get inception,WHS],[safe_get full,WHS],—,—,—"
close $fh
puts "  -> $csv"

puts "============================================================"
puts "  D1 compare done."
puts "  Open $OUT_DIR/d1_summary.txt for paper table."
puts "  Open $OUT_DIR/*_critical_path.pdf for paper figures."
puts "============================================================"
