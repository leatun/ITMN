`include "_parameter.v"

// ============================================================================
// LUT_Bank — nonlinear activation lookup block.
//
//   16-lane parallel activation LUTs (combinational, 1-cycle drive path):
//     - Silu_LUT      : x_in (Q4.11) → silu(x)         (M3, M7)
//     - Softplus_LUT  : x_in (Q4.11) → softplus(x)     (M5)
//     - Exp_LUT       : x_in (Q4.11) → exp(x)          (M6 DAB)
//   Each LUT type is 16 parallel copies because distributed-LUTRAM primitives
//   have 1 read port — 16 concurrent lookups require 16 physical instances.
//
//   RSqrt_ROM (8K × 16-bit BRAM-inferred):
//     - idx → data = 1/√(mean(x²))                     (RMSNorm)
//
// Tables are loaded from golden_all/{silu,softplus,exp}_lut.txt and
// golden_all/rsqrt_q97.txt at synth/sim time (inside each LUT sub-module).
//
// Separated from Memory_System (2026-07-13) so storage vs compute-LUT concerns
// live in different hierarchy blocks.
// ============================================================================

module LUT_Bank (
    // ---- Activation LUTs (256-bit = 16 × 16-bit lane) ----
    input  wire [255:0] silu_in_flat,
    input  wire [255:0] sp_in_flat,
    input  wire [255:0] exp_in_flat,
    output wire [255:0] silu_out_flat,
    output wire [255:0] sp_out_flat,
    output wire [255:0] exp_out_flat,

    // ---- RSqrt ROM ----
    input  wire [12:0] rsqrt_idx,
    output wire [15:0] rsqrt_data
);

    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : ACT_LANES
            Silu_LUT     u_silu (
                .x_in     (silu_in_flat [gi*16 +: 16]),
                .silu_out (silu_out_flat[gi*16 +: 16])
            );
            Softplus_LUT u_softplus (
                .x_in         (sp_in_flat  [gi*16 +: 16]),
                .softplus_out (sp_out_flat [gi*16 +: 16])
            );
            Exp_LUT      u_exp (
                .x_in    (exp_in_flat [gi*16 +: 16]),
                .exp_out (exp_out_flat[gi*16 +: 16])
            );
        end
    endgenerate

    RSqrt_ROM u_rsqrt_rom (
        .idx  (rsqrt_idx),
        .data (rsqrt_data)
    );

endmodule
