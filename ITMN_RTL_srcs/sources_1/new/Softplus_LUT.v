`include "_parameter.v"

// ============================================================================
// Softplus_LUT — combinational softplus(x) = log(1 + exp(x)) lookup.
//
//   Range covered : x_in ∈ [-8.0, +8.0) in Q4.11
//   Indexing      : identical scheme to Silu_LUT
//   Out of range  : x <= -8 → 0 (softplus → 0);   x >= +8 → x (softplus → x)
//
// Table loaded from golden_all/softplus_lut.txt.
// ============================================================================

module Softplus_LUT (
    input  signed [15:0] x_in,
    output signed [15:0] softplus_out
);

    localparam signed [15:0] LUT_LO    = -(16'sd1 << (`FRAC_BITS + 3));
    localparam        [3:0]  LUT_SHIFT = `FRAC_BITS - 4'd4;

    wire signed [15:0] x        = x_in;
    wire               in_range = (x >= LUT_LO) && (x < -LUT_LO);
    wire [7:0]         idx      = in_range ? ($signed(x - LUT_LO) >>> LUT_SHIFT)
                                           : 8'd0;

    reg signed [15:0] softplus_table [0:255];
    initial $readmemh("golden_all/softplus_lut.txt", softplus_table);

    wire signed [15:0] softplus_oor = (x < LUT_LO) ? 16'sd0 : x;
    assign softplus_out = in_range ? softplus_table[idx] : softplus_oor;

endmodule
