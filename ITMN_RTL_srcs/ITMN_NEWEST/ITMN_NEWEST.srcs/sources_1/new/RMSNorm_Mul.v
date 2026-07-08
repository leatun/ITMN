`include "_parameter.v"

// ============================================================================
// RMSNorm_Mul — 2-step normalize with 1 pipeline register stage.
//
// Computes:  out = sat16( (sat16((x * gamma) >>> FB) * S) >>> FB )
//
// Pipeline (1-cycle latency):
//   Cycle N   :  present (x_in, gamma_in, S_in)
//   Cycle N+1 :  p1_reg latched = sat16((x*γ) >>> FB),  S_reg latched = S
//                x_norm_out combinationally = sat16((p1_reg * S_reg) >>> FB)
//
// Replaces the inline `x_norm_fn` function in ITM_CONTROLLER. Tách 2 mult
// cascaded thành 2 cycle, register chen giữa, giảm critical path từ 2 DSP+sat
// nối tiếp xuống 1 DSP+sat per cycle. Functional byte-exact với bản cũ.
// ============================================================================
module RMSNorm_Mul (
    input  wire                  clk,
    input  wire signed [15:0]    x_in,
    input  wire signed [15:0]    gamma_in,
    input  wire signed [15:0]    S_in,
    output wire signed [15:0]    x_norm_out
);

    // ---- Stage 1: p1 = sat16((x_in * gamma_in) >>> FB), registered ----
    wire signed [31:0] p1_wide    = x_in * gamma_in;
    wire signed [31:0] p1_shifted = p1_wide >>> `FRAC_BITS;
    wire signed [15:0] p1_sat     = (p1_shifted >  32'sd32767) ? 16'sh7FFF :
                                    (p1_shifted < -32'sd32768) ? 16'sh8000 :
                                                                  p1_shifted[15:0];

    reg signed [15:0] p1_reg;
    reg signed [15:0] S_reg;
    always @(posedge clk) begin
        p1_reg <= p1_sat;
        S_reg  <= S_in;
    end

    // ---- Stage 2: out = sat16((p1_reg * S_reg) >>> FB), combinational ----
    wire signed [31:0] o_wide    = p1_reg * S_reg;
    wire signed [31:0] o_shifted = o_wide >>> `FRAC_BITS;
    assign x_norm_out = (o_shifted >  32'sd32767) ? 16'sh7FFF :
                        (o_shifted < -32'sd32768) ? 16'sh8000 :
                                                     o_shifted[15:0];

endmodule
