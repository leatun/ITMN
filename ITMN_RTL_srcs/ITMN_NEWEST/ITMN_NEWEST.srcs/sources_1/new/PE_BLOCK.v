`include "_parameter.v"

// ============================================================================
// PE_Array V2 - 16 Unified_PE instances with flexible input A.
//
// Modes:
//   a_is_vector = 0 (default, backward-compatible): broadcast in_A scalar to all PEs.
//                  Used for MAC reductions across input channels (P1, Inception, M1, M4).
//   a_is_vector = 1: use in_A_vec, each PE i gets lane i (16-bit slice).
//                    Used for element-wise ops (M2 depthwise conv1d, M7 gating).
//
// in_B is always vector (256-bit, 16 lanes).
// ============================================================================
module PE_Array (
    input clk, input rst,
    input clear_acc,
    input [1:0] op_mode,
    input signed [15:0] in_A,           // scalar broadcast (used when a_is_vector=0)
    input [255:0] in_A_vec,             // vector input (used when a_is_vector=1)
    input a_is_vector,
    input [255:0] in_B,
    output [255:0] out_vector
);
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : PE_CORE
            wire signed [15:0] pe_a_lane = a_is_vector ? in_A_vec[i*16 +: 16] : in_A;
            Unified_PE upe (
                .clk(clk), .reset(rst),
                .op_mode(op_mode), .clear_acc(clear_acc),
                .in_A(pe_a_lane), .in_B(in_B[i*16 +: 16]),
                .out_val(out_vector[i*16 +: 16])
            );
        end
    endgenerate
endmodule