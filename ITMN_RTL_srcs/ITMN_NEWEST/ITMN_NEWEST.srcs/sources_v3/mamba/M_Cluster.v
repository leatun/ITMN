`include "_parameter.v"

// ============================================================================
// M_Cluster — Mamba compute cluster (16-lane).
//   - 16 × Mamba_PE   (state-parallel datapath, lane = state index 0..15)
//   - 1  × H_RegFile  (on-chip h_state, 1R/1W, 256-bit word, 1-cycle latency)
//   - 1  × Reduce16   (combinational 16-way signed adder tree, y-reduction)
//
// Width = 256-bit (16 lanes × 16-bit) matches Memory_System cũ word size →
// reuse adapter-free.
//
// PE inputs vectorised (256-bit) and driven by external FSM_M. FSM is
// responsible for:
//   - addressing h_rf (rd_addr / wr_addr, indexed by channel c)
//   - choosing in_H source via h_from_rf:
//        h_from_rf=1 → in_H_vec = h_rd_data   (SSM update path)
//        h_from_rf=0 → in_H_vec = in_H_ext    (linear MAC / generic ops)
//   - asserting h_wr_en + h_wr_from_pe to capture out_vec back into H_RegFile.
//
// All PEs share op_mode + clear_acc (SIMD broadcast). y_reduce_out is the
// combinational sum of out_vec — used at the end of the y MAC chain to
// collapse 16 per-state partial sums into one scalar y[c].
// ============================================================================
module M_Cluster #(
    parameter H_ADDR_W = `H_ADDR_W,
    parameter H_DEPTH  = `H_DEPTH
) (
    input  wire                clk,
    input  wire                rst,

    // ---- PE control (broadcast to all 16 lanes) ----
    input  wire [2:0]          op_mode,
    input  wire                clear_acc,

    // ---- PE input vectors (16 lanes × 16-bit = 256-bit packed) ----
    input  wire [16*`DATA_W-1:0] in_W1_vec,
    input  wire [16*`DATA_W-1:0] in_H_ext,      // used when h_from_rf=0
    input  wire [16*`DATA_W-1:0] in_W2_vec,
    input  wire [16*`DATA_W-1:0] in_X_vec,

    // ---- H_RegFile control ----
    input  wire                  h_from_rf,     // 1: in_H = h_rd_data, else in_H_ext
    input  wire [H_ADDR_W-1:0]   h_rd_addr,
    input  wire                  h_wr_en,
    input  wire [H_ADDR_W-1:0]   h_wr_addr,
    input  wire                  h_wr_from_pe,  // 1: write out_vec, else h_wr_data_ext
    input  wire [16*`DATA_W-1:0] h_wr_data_ext, // external write data (e.g. zero init)

    // ---- Outputs ----
    output wire [16*`DATA_W-1:0] out_vec,          // 16 PE outputs (16-bit each)
    output wire [16*`ACC_W-1:0]  acc_raw_vec,      // 16 × 40-bit raw accumulators (for sum-of-squares)
    output wire [`DATA_W+4:0]    y_reduce_out,     // combinational reduce-sum (21-bit, sat outputs)
    output wire [16*`DATA_W-1:0] out_next_vec,     // Opt B: combinational next-outputs from all PEs
    output wire [16*`DATA_W-1:0] out_vec2,         // MUL2 secondary registered outputs
    output wire [16*`DATA_W-1:0] out_next_vec2     // MUL2 secondary combinational next-outputs
);

    // ---- H_RegFile instance ----
    wire [16*`DATA_W-1:0] h_rd_data;
    wire [16*`DATA_W-1:0] h_wr_data = h_wr_from_pe ? out_vec : h_wr_data_ext;

    H_RegFile #(
        .ADDR_W(H_ADDR_W),
        .DEPTH (H_DEPTH)
    ) u_hrf (
        .clk     (clk),
        .rst     (rst),
        .wr_en   (h_wr_en),
        .wr_addr (h_wr_addr),
        .wr_data (h_wr_data),
        .rd_addr (h_rd_addr),
        .rd_data (h_rd_data)
    );

    // ---- in_H vector mux ----
    wire [16*`DATA_W-1:0] in_H_vec = h_from_rf ? h_rd_data : in_H_ext;

    // ---- 16 × Mamba_PE ----
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : PE_LANE
            Mamba_PE u_pe (
                .clk          (clk),
                .rst          (rst),
                .op_mode      (op_mode),
                .clear_acc    (clear_acc),
                .in_W1        (in_W1_vec    [i*`DATA_W +: `DATA_W]),
                .in_H         (in_H_vec     [i*`DATA_W +: `DATA_W]),
                .in_W2        (in_W2_vec    [i*`DATA_W +: `DATA_W]),
                .in_X         (in_X_vec     [i*`DATA_W +: `DATA_W]),
                .out_val      (out_vec      [i*`DATA_W +: `DATA_W]),
                .acc_raw      (acc_raw_vec  [i*`ACC_W  +: `ACC_W ]),
                .out_next_exp (out_next_vec [i*`DATA_W +: `DATA_W]),
                .out_val2     (out_vec2     [i*`DATA_W +: `DATA_W]),
                .out_next2_exp(out_next_vec2[i*`DATA_W +: `DATA_W])
            );
        end
    endgenerate

    // ---- y reduction tree (combinational, 16-way) ----
    Reduce16 u_red (
        .in_vec (out_vec),
        .out_sum(y_reduce_out)
    );

endmodule
