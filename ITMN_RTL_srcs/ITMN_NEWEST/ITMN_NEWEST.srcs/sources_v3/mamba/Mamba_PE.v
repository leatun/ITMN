`include "_parameter.v"

// ============================================================================
// Mamba_PE — dual-multiplier PE specialized for SSM scan.
//
// Two DSP48E2 multipliers run in parallel:
//   m1 = in_W1 * in_H
//   m2 = in_W2 * in_X   (active only in SSM mode)
//
// 1-cycle latency: inputs on cycle N → out_val/acc_raw on cycle N+1.
//
// Modes (op_mode[2:0]):
//   IDLE : acc held, out=0
//   MAC  : acc <= clear_acc ? m1 : (acc + m1);   out <= sat16(acc_next >> FB)
//          (also used for SSM y-reduction:  y += h_new * C)
//   MUL  : acc <= m1;                            out <= sat16(m1 >> FB)
//   ADD  : acc <= ext(W1) + ext(H);              out <= sat16(W1 + H)
//   SSM  : acc <= m1 + m2;                       out <= sat16((m1+m2) >> FB)
//   MUL2 : acc <= m1_ext (don't-care);
//          out  <= sat16(m1 >> FB)  ;  out2 <= sat16(m2 >> FB)
//          → dual independent products in 1 cycle; enables M6 dt*A / dt*B
//            fusion and lets downstream consume registered outputs (breaks
//            the URAM→mult→Exp_LUT combinational chain that dominates fmax).
//
// Notes:
//   - clear_acc only acts in MAC mode; ignored elsewhere (SSM/MUL2 overwrite).
//   - When IDLE the m2 path is don't-care; Vivado will leave the DSP unused.
//   - sat16 saturates to int16 range.
//   - out_val2 / out_next2_exp are only meaningful in MUL2; other modes drive 0.
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
    output reg  signed [`ACC_W-1:0]  acc_raw,
    output wire signed [`DATA_W-1:0] out_next_exp, // Opt B: combinational next-output
    output reg  signed [`DATA_W-1:0] out_val2,     // MUL2 secondary output (m2>>FB)
    output wire signed [`DATA_W-1:0] out_next2_exp // Combinational next of out_val2
);

    // -----------------------------------------------------------------------
    // Combinational multiplies (Vivado infers DSP48E2 with internal regs if
    // upstream registers are present; here we register the products inside
    // acc_raw on the next clock edge).
    // -----------------------------------------------------------------------
    wire signed [2*`DATA_W-1:0] m1 = in_W1 * in_H;
    wire signed [2*`DATA_W-1:0] m2 = in_W2 * in_X;

    // Sign-extended versions for accumulator math
    wire signed [`ACC_W-1:0] m1_ext = {{(`ACC_W - 2*`DATA_W){m1[2*`DATA_W-1]}}, m1};
    wire signed [`ACC_W-1:0] m2_ext = {{(`ACC_W - 2*`DATA_W){m2[2*`DATA_W-1]}}, m2};
    wire signed [`ACC_W-1:0] w1_ext = {{(`ACC_W - `DATA_W){in_W1[`DATA_W-1]}}, in_W1};
    wire signed [`ACC_W-1:0] h_ext  = {{(`ACC_W - `DATA_W){in_H[`DATA_W-1]}},  in_H};
    wire signed [`ACC_W-1:0] sum_ssm = m1_ext + m2_ext;
    wire signed [`ACC_W-1:0] sum_add = w1_ext + h_ext;

    // Next-cycle accumulator value (mode-dependent)
    reg  signed [`ACC_W-1:0] acc_next;
    always @* begin
        case (op_mode)
            `MAMBA_PE_MAC:  acc_next = clear_acc ? m1_ext : (acc_raw + m1_ext);
            `MAMBA_PE_MAC2: acc_next = clear_acc ? sum_ssm : (acc_raw + sum_ssm);
            `MAMBA_PE_MUL:  acc_next = m1_ext;
            `MAMBA_PE_MUL2: acc_next = m1_ext;    // don't-care (out_val is primary)
            `MAMBA_PE_ADD:  acc_next = sum_add;
            `MAMBA_PE_SSM:  acc_next = sum_ssm;
            default:        acc_next = acc_raw;   // IDLE: hold
        endcase
    end

    // Saturating shift-right to int16 (in MAC/MUL/SSM modes shift by FB;
    // in ADD mode no shift — value is already int16-ish, just clip).
    function automatic signed [`DATA_W-1:0] sat16;
        input signed [`ACC_W-1:0] v;
        begin
            if      (v >  40'sd32767)  sat16 = 16'sh7FFF;
            else if (v < -40'sd32768)  sat16 = 16'sh8000;
            else                        sat16 = v[`DATA_W-1:0];
        end
    endfunction

    reg  signed [`DATA_W-1:0] out_next;
    always @* begin
        case (op_mode)
            `MAMBA_PE_MAC:  out_next = sat16(acc_next >>> `FRAC_BITS);
            `MAMBA_PE_MAC2: out_next = sat16(acc_next >>> `FRAC_BITS);
            `MAMBA_PE_MUL:  out_next = sat16(m1_ext   >>> `FRAC_BITS);
            `MAMBA_PE_MUL2: out_next = sat16(m1_ext   >>> `FRAC_BITS);
            `MAMBA_PE_ADD:  out_next = sat16(sum_add);
            `MAMBA_PE_SSM:  out_next = sat16(sum_ssm  >>> `FRAC_BITS);
            default:        out_next = 16'sd0;
        endcase
    end

    // MUL2 secondary output: sat16(m2 >> FB). Zero in every other mode so
    // downstream consumers can safely OR / mux without extra guards.
    reg signed [`DATA_W-1:0] out_next2;
    always @* begin
        case (op_mode)
            `MAMBA_PE_MUL2: out_next2 = sat16(m2_ext >>> `FRAC_BITS);
            default:        out_next2 = 16'sd0;
        endcase
    end

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
