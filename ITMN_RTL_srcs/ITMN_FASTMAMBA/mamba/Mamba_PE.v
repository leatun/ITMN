`include "_parameter.v"

// ============================================================================
// Mamba_PE — clean rewrite (2026-07-16).
//
// Physical storage: 1× acc_raw (40b) + 1× out_val (16b) + 1× out_val2 (16b) = 72b.
// No phantom acc_next reg — comb logic uses wires (ternary chain).
//
// Fabric adders (comb per cyc): 33b (m1+m2), 17b (W1+H), 40b (mac acc).
// vs original: 40b × 3 always-active (sum_ssm, sum_add, acc+m1) — save ~3 CARRY8/PE.
//
// MUX narrowed 7-way case → 3-way ternary → save ~80 LUT/PE.
//
// MAC branch coded canonical → Vivado can infer DSP48E2 MAC (mult+ALU+PREG).
//
// 1-cycle latency: inputs @ cyc N → out_val / acc_raw @ cyc N+1.
// Byte-exact preserved vs prior version.
//
// Modes (op_mode[2:0]):
//   IDLE : acc holds; out = 0
//   MAC  : acc <= clr? m1 : acc+m1       ; out <= sat16(acc_next >> FB)
//   MAC2 : acc <= clr? m1+m2 : acc+m1+m2 ; out <= sat16(acc_next >> FB)
//   MUL  : acc <= m1                     ; out <= sat16(m1 >> FB)
//   MUL2 : acc <= m1 (dc)                ; out <= sat16(m1>>FB); out2 <= sat16(m2>>FB)
//   ADD  : acc <= sxt(W1)+sxt(H)         ; out <= sat16(W1+H)   [NO shift]
//   SSM  : acc <= m1+m2                  ; out <= sat16((m1+m2) >> FB)
// ============================================================================
module Mamba_PE (
    input  wire                     clk,
    input  wire                     rst,
    input  wire [2:0]               op_mode,
    input  wire                     clear_acc,
    input  wire signed [`DATA_W-1:0] in_W1,
    input  wire signed [`DATA_W-1:0] in_H,
    input  wire signed [`DATA_W-1:0] in_W2,
    input  wire signed [`DATA_W-1:0] in_X,
    output reg  signed [`DATA_W-1:0] out_val,
    output reg  signed [`ACC_W-1:0]  acc_raw,      // ← ONLY 40b physical reg
    output wire signed [`DATA_W-1:0] out_next_exp,
    output reg  signed [`DATA_W-1:0] out_val2,
    output wire signed [`DATA_W-1:0] out_next2_exp
);

    // Mode decoders (1 LUT each)
    wire is_mac  = (op_mode == `MAMBA_PE_MAC );
    wire is_mac2 = (op_mode == `MAMBA_PE_MAC2);
    wire is_mul  = (op_mode == `MAMBA_PE_MUL );
    wire is_mul2 = (op_mode == `MAMBA_PE_MUL2);
    wire is_ssm  = (op_mode == `MAMBA_PE_SSM );
    wire is_add  = (op_mode == `MAMBA_PE_ADD );
    wire mac_en   = is_mac  | is_mac2;              // MAC accumulate
    wire load_en  = is_mul  | is_mul2 | is_ssm;     // direct load (no acc+)
    wire dual_sum = is_mac2 | is_ssm;               // addend = m1+m2

    // DSP mults — Vivado infers DSP48E2
    (* use_dsp = "yes" *)
    wire signed [2*`DATA_W-1:0] m1 = in_W1 * in_H;   // 32b product
    (* use_dsp = "yes" *)
    wire signed [2*`DATA_W-1:0] m2 = in_W2 * in_X;   // 32b product

    // Narrow inner sums — width = actual result, not 40b overkill
    wire signed [2*`DATA_W  :0] m12_sum =            // 33b: m1+m2 max range
        {m1[2*`DATA_W-1], m1} + {m2[2*`DATA_W-1], m2};
    wire signed [`DATA_W    :0] wh_sum =             // 17b: W1+H max range
        {in_W1[`DATA_W-1], in_W1} + {in_H[`DATA_W-1], in_H};

    // Sign-ext to 40b at consumption point
    wire signed [`ACC_W-1:0] m1_ext40  = {{(`ACC_W - 2*`DATA_W){m1[2*`DATA_W-1]}}, m1};
    wire signed [`ACC_W-1:0] m2_ext40  = {{(`ACC_W - 2*`DATA_W){m2[2*`DATA_W-1]}}, m2};
    wire signed [`ACC_W-1:0] m12_ext40 = {{(`ACC_W - (2*`DATA_W+1)){m12_sum[2*`DATA_W]}}, m12_sum};
    wire signed [`ACC_W-1:0] wh_ext40  = {{(`ACC_W - (`DATA_W+1)){wh_sum[`DATA_W]}}, wh_sum};

    // Addend for MAC/load path
    wire signed [`ACC_W-1:0] mac_addend = dual_sum ? m12_ext40 : m1_ext40;

    // Canonical MAC: (clr?0:p) + addend — Vivado sees P += M pattern
    wire signed [`ACC_W-1:0] mac_acc = (clear_acc ? {`ACC_W{1'b0}} : acc_raw) + mac_addend;

    // acc_next as WIRE (ternary chain) — no phantom reg
    wire signed [`ACC_W-1:0] acc_next =
        mac_en   ? mac_acc     :
        load_en  ? mac_addend  :
        is_add   ? wh_ext40    :
                   acc_raw;      // IDLE — hold

    // Sat function
    function automatic signed [`DATA_W-1:0] sat16;
        input signed [`ACC_W-1:0] v;
        begin
            if      (v >  40'sd32767)  sat16 = `SAT16_MAX;
            else if (v < -40'sd32768)  sat16 = `SAT16_MIN;
            else                        sat16 = v[`DATA_W-1:0];
        end
    endfunction

    // out_next: mode-specific shift
    wire signed [`DATA_W-1:0] out_next =
        mac_en   ? sat16(acc_next   >>> `FRAC_BITS) :
        load_en  ? sat16(mac_addend >>> `FRAC_BITS) :
        is_add   ? sat16(wh_ext40) :
                   {`DATA_W{1'b0}};

    // MUL2 secondary output
    wire signed [`DATA_W-1:0] out_next2 =
        is_mul2 ? sat16(m2_ext40 >>> `FRAC_BITS) : {`DATA_W{1'b0}};

    // Register update — 1× 40b (acc_raw) + 2× 16b (out_val, out_val2)
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_raw  <= {`ACC_W{1'b0}};
            out_val  <= {`DATA_W{1'b0}};
            out_val2 <= {`DATA_W{1'b0}};
        end else begin
            acc_raw  <= acc_next;
            out_val  <= out_next;
            out_val2 <= out_next2;
        end
    end

    assign out_next_exp  = out_next;
    assign out_next2_exp = out_next2;

endmodule
