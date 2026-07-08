`timescale 1ns / 1ps

// ============================================================================
// RSqrt_ROM — 8192-entry × 16-bit table holding the RMSNorm rsqrt result.
//
//   ROM[m] = round(sqrt(2^(1+N)) * SCALE / sqrt(m))   for m in 0..8191
//   For FB=11, N=6:  ROM[128] ≈ 2048 = rsqrt(1.0) in Q4.11.
//   Indexed by norm_rom_idx (computed in controller from norm_sq_acc).
//
//   Inferred as BRAM by Vivado (default ram_style for 8K×16). Output is
//   combinational w.r.t. addr — no clock port, no register stage. Used in
//   exactly one place: S_NORM_M1[AB]_MEAN where `norm_S_reg` latches the
//   value once per timestep.
//
//   To force distributed LUT implementation (frees 4 BRAM, adds ~2500 LUT),
//   wrap the array decl with `(* rom_style = "distributed" *)`.
// ============================================================================

module RSqrt_ROM (
    input  [12:0] idx,
    output [15:0] data
);

    reg [15:0] rom [0:8191];
    initial $readmemh("golden_all/rsqrt_q97.txt", rom);

    assign data = rom[idx];

endmodule
