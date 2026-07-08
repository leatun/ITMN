`include "_parameter.v"

// ============================================================================
// Silu_LUT — combinational SiLU(x) lookup, 256 entries × 16-bit Q4.11.
//
//   Range covered : x_in ∈ [-8.0, +8.0) in Q4.11 (= int range [-16384, 16384))
//   Resolution    : 16 integer units per entry (≈ 0.0078 float per entry)
//   Index         : idx = (x_in - LUT_LO) >>> LUT_SHIFT
//                       LUT_LO    = -(1 << (FRAC_BITS+3)) = -16384
//                       LUT_SHIFT =  FRAC_BITS - 4       =  7
//
//   Out of range  : x <= -8 → 0;   x >= +8 → x   (silu(x) → x for large x)
//
// Table values loaded at synth/sim init from golden_all/silu_lut.txt.
// ============================================================================

module Silu_LUT (
    input  signed [15:0] x_in,
    output signed [15:0] silu_out
);

    localparam signed [15:0] LUT_LO    = -(16'sd1 << (`FRAC_BITS + 3));
    localparam        [3:0]  LUT_SHIFT = `FRAC_BITS - 4'd4;

    wire signed [15:0] x        = x_in;
    wire               in_range = (x >= LUT_LO) && (x < -LUT_LO);
    wire [7:0]         idx      = in_range ? ($signed(x - LUT_LO) >>> LUT_SHIFT)
                                           : 8'd0;

    reg signed [15:0] silu_table [0:255];
    initial $readmemh("golden_all/silu_lut.txt", silu_table);

    wire signed [15:0] silu_oor = (x < LUT_LO) ? 16'sd0 : x;
    assign silu_out = in_range ? silu_table[idx] : silu_oor;

endmodule
