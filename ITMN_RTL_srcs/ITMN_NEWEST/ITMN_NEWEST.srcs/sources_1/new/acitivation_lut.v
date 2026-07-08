`include "_parameter.v"

// ============================================================================
// Activation_LUT - combinational lookup tables for SiLU and Softplus.
//
// Both functions take Q9.7 input (16-bit signed) and produce Q9.7 output.
//
// LUT design: 256 entries indexed by input range [-8.0, +8.0).
//   For input x in Q9.7, |x| ? 8 means |x_int| ? 1024.
//   Within range, index = (x_int + 1024) >> 3   (gives 0..255)
//   Each LUT entry is precomputed value of the function at that input.
//
// Out-of-range behavior:
//   SiLU:
//     x ? -8:  silu(x) ? 0  (sigmoid(x) ? 0, x�0 = 0; small negative for x near 0)
//     x ?  8:  silu(x) ? x  (sigmoid(x) ? 1, x�1 = x)
//   Softplus:
//     x ? -8:  softplus(x) ? exp(x) ? 0
//     x ?  8:  softplus(x) ? x  (since log(1+e^x) ? x for large x)
//
// LUT values precomputed in Python and pasted below as initial array.
// ============================================================================

module Activation_LUT (
    input  signed [15:0] x_in,
    output signed [15:0] silu_out,
    output signed [15:0] softplus_out,
    output signed [15:0] exp_out
);

    // LUT range: float [-8, +8)  →  int [-(8<<FRAC_BITS), 8<<FRAC_BITS)
    // LUT_SHIFT = FRAC_BITS - 4,  index = (x - LUT_LO) >> LUT_SHIFT  ∈ [0..255]
    localparam signed [15:0] LUT_LO    = -(16'sd1 << (`FRAC_BITS + 3));
    localparam        [3:0]  LUT_SHIFT = `FRAC_BITS - 4'd4;
    wire signed [15:0] x = x_in;
    wire in_range = (x >= LUT_LO) && (x < -LUT_LO);
    wire [7:0] idx = in_range ? ($signed(x - LUT_LO) >>> LUT_SHIFT) : 8'd0;
    
    // SiLU LUT (256 entries)
    reg signed [15:0] silu_table [0:255];
    // Softplus LUT (256 entries)
    reg signed [15:0] softplus_table [0:255];
    // Exp LUT (256 entries) - for SSM scan dA = exp(delta * A)
    // Range: input is delta*A which is non-positive since A=-exp(A_log) is negative,
    // delta is positive (after softplus). Practical range delta*A in [-30, 0] roughly.
    // LUT indexed by [-8, +8) as well. For x < -8: exp(x) ? 0. For x > 0: exp(x) > 1, capped.
    reg signed [15:0] exp_table [0:255];
    
    initial begin
        $readmemh("golden_all/silu_lut.txt", silu_table);
        $readmemh("golden_all/softplus_lut.txt", softplus_table);
        $readmemh("golden_all/exp_lut.txt", exp_table);
        $display("[LUT] FRAC_BITS=%0d LUT_LO=%0d LUT_SHIFT=%0d", `FRAC_BITS, LUT_LO, LUT_SHIFT);
    end
    
    wire signed [15:0] silu_lut_val = silu_table[idx];
    wire signed [15:0] softplus_lut_val = softplus_table[idx];
    wire signed [15:0] exp_lut_val = exp_table[idx];
    
    // Out-of-range fallbacks
    wire signed [15:0] silu_oor     = (x < LUT_LO) ? 16'sd0 : x;
    wire signed [15:0] softplus_oor = (x < LUT_LO) ? 16'sd0 : x;
    wire signed [15:0] exp_oor      = (x < LUT_LO) ? 16'sd0 : 16'sh7FFF;
    
    assign silu_out     = in_range ? silu_lut_val     : silu_oor;
    assign softplus_out = in_range ? softplus_lut_val : softplus_oor;
    assign exp_out      = in_range ? exp_lut_val      : exp_oor;

endmodule