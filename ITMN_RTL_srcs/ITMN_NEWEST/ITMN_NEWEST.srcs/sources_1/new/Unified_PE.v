`include "_parameter.v"

// ============================================================================
// Unified_PE - 40-bit accumulator, deferred shift, clear_acc captures first MAC.
//
// Timing:
//   Cycle N: in_A, in_B, clear_acc present (combinational inputs to PE)
//   @posedge cycle N: acc_raw updates based on current inputs
//                     out_val = saturate(acc_raw_new >>> FRAC_BITS)
//
// clear_acc=1 + MAC mode: acc_raw <= in_A * in_B (start new chain with 1st product)
// clear_acc=0 + MAC mode: acc_raw <= acc_raw + in_A * in_B
// ============================================================================
module Unified_PE (
    input clk,
    input reset,
    input [1:0] op_mode,
    input clear_acc,
    input signed [15:0] in_A,
    input signed [15:0] in_B,
    output reg signed [15:0] out_val
);

    wire signed [31:0] mult = in_A * in_B;
    wire signed [39:0] mult_ext = {{8{mult[31]}}, mult};
    reg  signed [39:0] acc_raw;
    wire signed [39:0] acc_new = clear_acc ? mult_ext : acc_raw + mult_ext;

    function signed [15:0] sat16;
        input signed [39:0] v;
        if      (v > 40'sd32767)  sat16 = 16'sh7FFF;
        else if (v < -40'sd32768) sat16 = 16'sh8000;
        else                       sat16 = v[15:0];
    endfunction

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            acc_raw <= 40'sd0;
            out_val <= 16'sd0;
        end else begin
            case (op_mode)
                `MODE_MAC: begin
                    acc_raw <= acc_new;
                    out_val <= sat16(acc_new >>> `FRAC_BITS);
                end
                `MODE_MUL: begin
                    acc_raw <= mult_ext;
                    out_val <= sat16(mult_ext >>> `FRAC_BITS);
                end
                `MODE_ADD: begin
                    acc_raw <= {{24{in_A[15]}}, in_A} + {{24{in_B[15]}}, in_B};
                    out_val <= sat16({{24{in_A[15]}}, in_A} + {{24{in_B[15]}}, in_B});
                end
                default: begin
                    acc_raw <= 40'sd0;
                    out_val <= 16'sd0;
                end
            endcase
        end
    end

endmodule