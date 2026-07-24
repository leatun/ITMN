`include "_parameter.v"

// ============================================================================
// Exp_LUT — combinational exp(x) lookup for Mamba SSM scan (dA = exp(δ·A)).
//
//   Range covered : x_in ∈ [-8.0, +8.0) in Q4.11
//   Indexing      : identical scheme to Silu_LUT
//   Out of range  : x <= -8 → 0 (exp underflow);   x >= +8 → 32767 (sat Q4.11)
//
// Practical input range: δ·A is non-positive (A = -exp(A_log) is negative,
// δ ≥ 0 after softplus), so most accesses fall in [-30, 0]; the LUT covers
// the relevant near-zero portion with finer resolution.
//
// Table loaded from golden_all/exp_lut.txt.
// ============================================================================

module Exp_LUT (
    input  signed [15:0] x_in,
    output signed [15:0] exp_out
);

    localparam signed [15:0] LUT_LO    = -(16'sd1 << (`FRAC_BITS + 3));
    localparam        [3:0]  LUT_SHIFT = `FRAC_BITS - 4'd4;

    wire signed [15:0] x        = x_in;
    wire               in_range = (x >= LUT_LO) && (x < -LUT_LO);
    wire [7:0]         idx      = in_range ? ($signed(x - LUT_LO) >>> LUT_SHIFT)
                                           : 8'd0;

    reg signed [15:0] exp_table [0:255];
    initial $readmemh({`LUT_DIR, "/exp_lut.txt"}, exp_table);

    wire signed [15:0] exp_oor = (x < LUT_LO) ? 16'sd0 : 16'sh7FFF;
    assign exp_out = in_range ? exp_table[idx] : exp_oor;

endmodule
