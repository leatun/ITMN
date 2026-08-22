`include "_parameter.v"

// ============================================================================
// Reduce16Wide — sum 16 × 40-bit signed values, combinational.
//
// Used for RMSNorm sum-of-squares: M_Cluster runs MAC mode with W1=H=x,
// each lane k accumulates Σ_{c_grp} x[c_grp*16+k, t]². acc_raw_vec[k] holds
// 40-bit signed per-lane sum. Reduce16Wide sums 16 lanes to get the full
// sum_d (sum over all d_model channels) without any sat/shift precision loss.
//
// Bit growth: 16 × signed40 → signed44 worst-case. Output extended to 45-bit
// signed for headroom.
//
// Latency: 0 cycle. Tree depth = 4 adders.
//   stage0: 16 in40 → 8 in41
//   stage1:  8 in41 → 4 in42
//   stage2:  4 in42 → 2 in43
//   stage3:  2 in43 → 1 in44  (sign-extended → 45-bit out)
// ============================================================================
module Reduce16Wide (
    input  wire signed [16*`ACC_W-1:0] in_vec,
    output wire signed [`ACC_W+4:0]    out_sum     // 45-bit signed
);

    wire signed [`ACC_W-1:0] lane [0:15];
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : UNPACK
            assign lane[i] = in_vec[i*`ACC_W +: `ACC_W];
        end
    endgenerate

    wire signed [`ACC_W:0]   s0_0 = lane[0]  + lane[1];
    wire signed [`ACC_W:0]   s0_1 = lane[2]  + lane[3];
    wire signed [`ACC_W:0]   s0_2 = lane[4]  + lane[5];
    wire signed [`ACC_W:0]   s0_3 = lane[6]  + lane[7];
    wire signed [`ACC_W:0]   s0_4 = lane[8]  + lane[9];
    wire signed [`ACC_W:0]   s0_5 = lane[10] + lane[11];
    wire signed [`ACC_W:0]   s0_6 = lane[12] + lane[13];
    wire signed [`ACC_W:0]   s0_7 = lane[14] + lane[15];

    wire signed [`ACC_W+1:0] s1_0 = s0_0 + s0_1;
    wire signed [`ACC_W+1:0] s1_1 = s0_2 + s0_3;
    wire signed [`ACC_W+1:0] s1_2 = s0_4 + s0_5;
    wire signed [`ACC_W+1:0] s1_3 = s0_6 + s0_7;

    wire signed [`ACC_W+2:0] s2_0 = s1_0 + s1_1;
    wire signed [`ACC_W+2:0] s2_1 = s1_2 + s1_3;

    wire signed [`ACC_W+3:0] s3_0 = s2_0 + s2_1;

    assign out_sum = {s3_0[`ACC_W+3], s3_0};

endmodule
