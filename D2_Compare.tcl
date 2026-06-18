# ============================================================================
# D2_Compare.tcl — Open ITM_Top_v3 (E2E full) checkpoint, dump comparison
# artifacts to reports/d2_compare/.
#
# Reuses D1 numbers (Mamba_Top + Inception_Top + ITM_Top_v2 full kernel) as
# baseline and reports the overhead of the new encoder + GAP + FC head.
#
# Run from Vivado tcl shell (Tcl Console or batch):
#   source D2_Compare.tcl
#
# Produces (no PDF schematic export):
#   reports/d2_compare/
#     v3_util_hier.rpt          hierarchical utilization for ITM_Top_v3
#     v3_critical_path.rpt      worst-case timing path
#     d2_summary.txt            side-by-side D2 (E2E) vs D1 (kernel) table
#     d2_summary.csv            same table, Excel/LaTeX import
#
# Sequencing assumption: E2E_OOC.tcl + Mamba_OOC.tcl + Inception_OOC.tcl have
# already run.  The Mamba/Inception/v2-full numbers come from the fixed table
# at the bottom (= the previously-saved d1_summary.csv).
# ============================================================================

set OUT_DIR reports/d2_compare
file mkdir $OUT_DIR

set V3_DCP reports/v3/itm_top_v3_routed.dcp

array set R {}

proc grab_int {report regex} {
    if {[regexp -line $regex $report -> m]} { return $m }
    return 0
}

# ---------------------------------------------------------------------------
# Helper: per-design dump (hierarchical util + critical path text)
# ---------------------------------------------------------------------------
proc dump_v3 {dcp_path out_dir} {
    global R
    puts "============================================================"
    puts "  Loading v3 (E2E full) checkpoint: $dcp_path"
    puts "============================================================"
    open_checkpoint $dcp_path

    # 1. Hierarchical utilization (encoder vs ITM body vs head attribution)
    set util_rpt $out_dir/v3_util_hier.rpt
    report_utilization -hierarchical -hierarchical_depth 4 -file $util_rpt
    puts "  -> $util_rpt"

    # 2. Capture top-level numbers
    set u_flat [report_utilization -return_string]
    set R(v3,LUT)   [grab_int $u_flat {\|\s*CLB LUTs\s+\|\s*(\d+)}]
    set R(v3,REG)   [grab_int $u_flat {\|\s*CLB Registers\s+\|\s*(\d+)}]
    set R(v3,CARRY) [grab_int $u_flat {\|\s*CARRY8\s+\|\s*(\d+)}]
    set R(v3,BRAM)  [grab_int $u_flat {\|\s*Block RAM Tile\s+\|\s*(\d+)}]
    set R(v3,URAM)  [grab_int $u_flat {\|\s*URAM\s+\|\s*(\d+)}]
    set R(v3,DSP)   [grab_int $u_flat {\|\s*DSPs\s+\|\s*(\d+)}]

    # 3. Critical path — text report only (no PDF schematic export)
    set path_rpt $out_dir/v3_critical_path.rpt
    report_timing -max_paths 1 -nworst 1 -delay_type max -path_type full \
        -input_pins -file $path_rpt
    puts "  -> $path_rpt"

    # 4. Capture WNS / WHS from timing summary
    set t_flat [report_timing_summary -return_string]
    if {[regexp {(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\d+\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)} \
            $t_flat -> wns tns whs ths]} {
        set R(v3,WNS) $wns
        set R(v3,WHS) $whs
    } else {
        set R(v3,WNS) "n/a"
        set R(v3,WHS) "n/a"
    }

    close_design
}

# ---------------------------------------------------------------------------
# Run on v3
# ---------------------------------------------------------------------------
dump_v3 $V3_DCP $OUT_DIR

# ---------------------------------------------------------------------------
# Fixed D1 reference numbers (from reports/d1_compare/d1_summary.csv,
# c0d2d72 lock-in).  Keep in sync if D1 is re-synthesised.
# ---------------------------------------------------------------------------
set R(mamba,LUT)   7513
set R(mamba,REG)   3945
set R(mamba,CARRY) 420
set R(mamba,BRAM)  118
set R(mamba,URAM)  40
set R(mamba,DSP)   39
set R(mamba,WNS)   0.710
set R(mamba,WHS)   0.095

set R(incept,LUT)   5346
set R(incept,REG)   2103
set R(incept,CARRY) 379
set R(incept,BRAM)  118
set R(incept,URAM)  40
set R(incept,DSP)   37
set R(incept,WNS)   1.973
set R(incept,WHS)   0.096

set R(v2,LUT)   10491
set R(v2,REG)   4504
set R(v2,CARRY) 0
set R(v2,BRAM)  118
set R(v2,URAM)  40
set R(v2,DSP)   59
set R(v2,WNS)   0.646
set R(v2,WHS)   0.089

proc safe_get {key} { global R; if {[info exists R($key)]} { return $R($key) } else { return 0 } }

# ---------------------------------------------------------------------------
# Side-by-side summary text
# ---------------------------------------------------------------------------
set txt $OUT_DIR/d2_summary.txt
set fh [open $txt w]
puts $fh "============================================================"
puts $fh "  D2 vs D1 — E2E full (encoder + 5 ITM + GAP + FC) vs"
puts $fh "             kernel-only references"
puts $fh "  Target: KV260 / xck26-sfvc784-2LV-c, OOC synth, clk=10 ns"
puts $fh "============================================================"
puts $fh ""
puts $fh "Resource     |    Mamba |  Incept. |  v2 Full |  v3 E2E  | v3 - v2 (E2E overhead)"
puts $fh "-------------+----------+----------+----------+----------+-----------------------"
foreach {key fmt} {LUT %d REG %d CARRY %d BRAM %d URAM %d DSP %d WNS %.3f WHS %.3f} {
    set m  [safe_get mamba,$key]
    set i  [safe_get incept,$key]
    set v2 [safe_get v2,$key]
    set v3 [safe_get v3,$key]
    set delta_str "—"
    if {$fmt eq "%d"} {
        set delta [expr {$v3 - $v2}]
        set pct   [expr {$v2 > 0 ? round(100.0 * $delta / $v2) : 0}]
        set delta_str [format "%+d  (%+d%%)" $delta $pct]
    } else {
        # Floating WNS/WHS — print delta in ns
        if {[catch {expr {$v3 - $v2}} d_ns] == 0} {
            set delta_str [format "%+.3f ns" $d_ns]
        }
    }
    puts $fh [format "%-12s | %8s | %8s | %8s | %8s | %s" \
        $key [format $fmt $m] [format $fmt $i] [format $fmt $v2] [format $fmt $v3] $delta_str]
}
puts $fh ""
puts $fh "Notes:"
puts $fh "  - v2 Full = ITM_Top_v2 (5 ITM blocks only, no encoder/GAP/FC; D1 lock-in c0d2d72)"
puts $fh "  - v3 E2E  = ITM_Top_v3 (v1 base + encoder + GAP + FC integrated)"
puts $fh "  - v3 reuses PE_Array / Memory_System / RMSNorm_Mul for encoder MAC"
puts $fh "  - GAP uses dedicated 24-bit accumulators (gap_sum[8][16])"
puts $fh "  - FC uses dedicated 40-bit accumulator (fc_acc) + 16-lane parallel reduce"
puts $fh "    (likely adds ~16 DSP and a 16-input adder tree to critical path)"
puts $fh "  - Const_Storage ram_const expanded 64→128 entries (encoder+FC bias)"
close $fh
puts "  -> $txt"

# ---------------------------------------------------------------------------
# CSV (Excel / LaTeX import)
# ---------------------------------------------------------------------------
set csv $OUT_DIR/d2_summary.csv
set fh [open $csv w]
puts $fh "Resource,Mamba_Top,Inception_Top,v2_Full_ITM,v3_E2E,Overhead_abs,Overhead_pct"
foreach key {LUT REG CARRY BRAM URAM DSP} {
    set m  [safe_get mamba,$key]
    set i  [safe_get incept,$key]
    set v2 [safe_get v2,$key]
    set v3 [safe_get v3,$key]
    set d  [expr {$v3 - $v2}]
    set p  [expr {$v2 > 0 ? round(100.0 * $d / $v2) : 0}]
    puts $fh "$key,$m,$i,$v2,$v3,$d,$p%"
}
puts $fh "WNS_ns,[safe_get mamba,WNS],[safe_get incept,WNS],[safe_get v2,WNS],[safe_get v3,WNS],—,—"
puts $fh "WHS_ns,[safe_get mamba,WHS],[safe_get incept,WHS],[safe_get v2,WHS],[safe_get v3,WHS],—,—"
close $fh
puts "  -> $csv"

puts "============================================================"
puts "  D2 compare done."
puts "  Open $OUT_DIR/d2_summary.txt for the report table."
puts "  Hier util  : $OUT_DIR/v3_util_hier.rpt"
puts "  Crit path  : $OUT_DIR/v3_critical_path.rpt"
puts "============================================================"
