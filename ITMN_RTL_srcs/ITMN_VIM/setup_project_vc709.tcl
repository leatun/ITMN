# ============================================================================
# setup_project_vc709.tcl
#   Create Vivado project targeting Xilinx Virtex-7 VC709 (xc7vx690tffg1761-2).
#   Matches Vim (bidirectional Mamba1) - HW baseline VC709 for fair compare baseline (VC709).
#   No URAM (Virtex-7 doesn't have URAM) - all memories use BRAM36 inference.
#
# Usage from Vivado Tcl Console:
#   cd C:/Users/ADMIN/Downloads/ITMN_MASTER/ITMN_RTL_srcs/ITMN_LIGHTMAMBA
#   source setup_project_vc709.tcl
#
# Then run OOC synth via run_synth_only_vc709.tcl.
# ============================================================================

set proj_name "ITMN_VIM_VC709"
set proj_dir  [file normalize [file dirname [info script]]]
set part      "xc7vx690tffg1761-2"

# Cleanup old project if exists
if {[file exists "${proj_dir}/${proj_name}.xpr"]} {
    puts "Removing existing project ${proj_name}..."
    file delete -force "${proj_dir}/${proj_name}.cache"
    file delete -force "${proj_dir}/${proj_name}.hw"
    file delete -force "${proj_dir}/${proj_name}.ip_user_files"
    file delete -force "${proj_dir}/${proj_name}.runs"
    file delete -force "${proj_dir}/${proj_name}.sim"
    file delete -force "${proj_dir}/${proj_name}.srcs"
    file delete -force "${proj_dir}/${proj_name}.xpr"
}

create_project ${proj_name} ${proj_dir} -part ${part} -force
set_property target_language Verilog [current_project]

# Add sources (all .v files under common/, mamba/, top/)
foreach dir {common mamba top} {
    foreach f [glob -nocomplain "${proj_dir}/${dir}/*.v"] {
        add_files -norecurse -fileset sources_1 $f
    }
}

# Add TB files (default: Mamba2-130M TB)
foreach f [glob -nocomplain "${proj_dir}/tb/*.v"] {
    add_files -norecurse -fileset sim_1 $f
}

# Add XDC constraints
add_files -norecurse -fileset constrs_1 "${proj_dir}/common/mamba_top_mamba2.xdc"

set_property top Mamba_Top      [current_fileset]
# Default sim top: 130M. Change to _370M, _780M, _1_3B, _2_7B as needed.
set_property top tb_Mamba_Top_VIM_TI [get_filesets sim_1]

# Include path so `include "_parameter.v"` resolves
set_property include_dirs "${proj_dir}/common" [current_fileset]
set_property include_dirs "${proj_dir}/common" [get_filesets sim_1]

# OOC synth mode
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-mode out_of_context} -objects [get_runs synth_1]

puts ""
puts "============================================================"
puts "  Project ${proj_name} created for VC709 (${part})"
puts "  Sources: common/*.v, mamba/*.v, top/*.v"
puts "  Testbench: tb_Mamba_Top_MAMBA2 (Vim-Ti default; switch to _S/_B for larger variants)"
puts "  Ready for: launch_runs synth_1 -jobs 8"
puts "============================================================"
