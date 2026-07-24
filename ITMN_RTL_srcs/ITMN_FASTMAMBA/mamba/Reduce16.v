`include "_parameter.v"

// ============================================================================
// Reduce16 — 16-lane signed adder tree, combinational.
//
// Used after Mamba M6 SSM scan: y_partial[c] = Σ_{s=0..15} h_new[s,c] * C[s]
// Cluster is 16-lane, lane = state index s. Each MAC cycle produces 16 int16
// lane outputs (one per state). Reduce16 collapses them to a single y[c] in
// the same cycle as the final MAC writeback.
//
// Bit growth: 16 × signed16 → signed20 worst-case. We extend to 21 bits for
// 1-bit headroom (downstream sat16 / sat_add).
//
// Latency: 0 cycle (combinational). Tree depth = 4 adders.
//   stage0: 16 in16 → 8 in17
//   stage1:  8 in17 → 4 in18
//   stage2:  4 in18 → 2 in19
//   stage3:  2 in19 → 1 in20  (sign-extended → 21-bit out)
//
// 4-level 16-bit add ≈ 4 × 0.5 ns ≈ 2 ns — within 10 ns period budget.
// ============================================================================
module Reduce16 (
    input  wire signed [16*`DATA_W-1:0] in_vec,         // 16 lanes packed
    output wire signed [`DATA_W+4:0]    out_sum         // 21-bit signed sum
);

    wire signed [`DATA_W-1:0] lane [0:15];
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : UNPACK
            assign lane[i] = in_vec[i*`DATA_W +: `DATA_W];
        end
    endgenerate

    // Stage 0: 16 → 8 (17-bit)
    wire signed [`DATA_W:0]   s0_0 = lane[0]  + lane[1];
    wire signed [`DATA_W:0]   s0_1 = lane[2]  + lane[3];
    wire signed [`DATA_W:0]   s0_2 = lane[4]  + lane[5];
    wire signed [`DATA_W:0]   s0_3 = lane[6]  + lane[7];
    wire signed [`DATA_W:0]   s0_4 = lane[8]  + lane[9];
    wire signed [`DATA_W:0]   s0_5 = lane[10] + lane[11];
    wire signed [`DATA_W:0]   s0_6 = lane[12] + lane[13];
    wire signed [`DATA_W:0]   s0_7 = lane[14] + lane[15];

    // Stage 1: 8 → 4 (18-bit)
    wire signed [`DATA_W+1:0] s1_0 = s0_0 + s0_1;
    wire signed [`DATA_W+1:0] s1_1 = s0_2 + s0_3;
    wire signed [`DATA_W+1:0] s1_2 = s0_4 + s0_5;
    wire signed [`DATA_W+1:0] s1_3 = s0_6 + s0_7;

    // Stage 2: 4 → 2 (19-bit)
    wire signed [`DATA_W+2:0] s2_0 = s1_0 + s1_1;
    wire signed [`DATA_W+2:0] s2_1 = s1_2 + s1_3;

    // Stage 3: 2 → 1 (20-bit)
    wire signed [`DATA_W+3:0] s3_0 = s2_0 + s2_1;

    assign out_sum = {s3_0[`DATA_W+3], s3_0};        // sign-extend → 21-bit

endmodule
