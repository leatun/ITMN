`include "_parameter.v"
`include "_block_params.v"

// ============================================================================
// ITM_Top V10 - Parameterized ITM Block Controller (Block-4 inception fix)
//
// Supports all 5 ITM block configs:
//   block 0,1: T=1000, CH_IN=4 (d_in=64),  CH_M=8  (d_inner=128), DT_RANK=4
//   block 2,3: T=500,  CH_IN=4,             CH_M=8,                DT_RANK=4
//   block 4:   T=250,  CH_IN=8 (d_in=128),  CH_M=16 (d_inner=256), DT_RANK=8
//
// Block-4 inception fix:
//   dim = d_out/4.  Blocks 0-3: dim=16 (1 output word/branch/t).
//                   Block  4:   dim=32 (2 output words/branch/t).
//   c_grp_br loops 0..br_grp_last for each branch before advancing branch_id.
//   Weight/data addresses generalised.  fin_read_base generalised to 8 groups.
// ============================================================================

module ITM_Top (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done_phase1,
    output reg         done_inception,
    output reg         done_mamba,
    output reg         done_all,

    input  wire [9:0]  T_MAX,
    input  wire [3:0]  CH_IN,
    input  wire [3:0]  CH_OUT,
    input  wire [3:0]  CH_M,
    input  wire [3:0]  DT_RANK,

    input  wire        dma_write_en,
    input  wire [1:0]  dma_target,
    input  wire [14:0] dma_addr,
    input  wire [255:0] dma_wdata,
    output wire        dma_ready
);

    assign dma_ready = ~(dma_write_en & (dma_target == 2'd2 | dma_target == 2'd3) & m_we);

    // ---- Derived parameters ----
    wire [9:0]  t_last       = T_MAX - 10'd1;
    wire [3:0]  ch_in_last   = CH_IN  - 4'd1;
    wire [3:0]  ch_out_last  = CH_OUT - 4'd1;
    wire [3:0]  ch_m_last    = CH_M   - 4'd1;
    wire [3:0]  dt_last      = DT_RANK - 4'd1;
    wire [7:0]  d_in         = {CH_IN,  4'b0};   // P1 input channels
    wire [7:0]  d_out        = {CH_OUT, 4'b0};   // P1 output channels
    wire [7:0]  d_in_last    = d_in - 8'd1;
    wire [7:0]  ch_m_actual  = (CH_M == 4'd0) ? 8'd16 : {4'd0, CH_M};
    wire [7:0]  d_inner      = {ch_m_actual[3:0], 4'b0};
    wire [7:0]  d_inner_last = d_inner - 8'd1;

    wire [14:0] w_dw_size    = {7'd0, ch_m_actual} * 15'd4;               // CH_M * 4
    wire [14:0] W_XPROJ_BASE_W = W_M_DW_BASE + w_dw_size;
    wire [14:0] xproj_sz    = 15'd3 * {7'd0, d_inner};                   // 3 * d_inner
    wire [14:0] W_DTPROJ_BASE  = W_XPROJ_BASE_W + xproj_sz;
    wire [14:0] dtproj_sz   = {7'd0, ch_m_actual} * {11'd0, DT_RANK};    // CH_M * DT_RANK
    wire [14:0] W_ALOG_BASE    = W_DTPROJ_BASE + dtproj_sz;
    wire [14:0] alog_sz     = {7'd0, ch_m_actual} * 15'd16;              // CH_M * d_state(16)
    wire [14:0] W_DPARAM_BASE  = W_ALOG_BASE + alog_sz;
    wire [14:0] dparam_sz   = {7'd0, ch_m_actual};                       // CH_M
    wire [14:0] W_OUTPROJ_BASE = W_DPARAM_BASE + dparam_sz;

    // Address stride precomputes (registered combinational)
    reg  [14:0] t_stride_in;
    reg  [14:0] t_stride_m;
    reg  [14:0] t_stride_out;
    reg  [14:0] t_stride_xp;
    always @(*) begin
        t_stride_in  = {5'd0, t_cnt} * {11'd0, CH_IN};
        t_stride_m   = {5'd0, t_cnt} * {7'd0,  ch_m_actual};
        t_stride_out = {5'd0, t_cnt} * {11'd0, CH_OUT};
        t_stride_xp  = {5'd0, t_cnt} * 15'd3;
    end

    // Fixed RAM base addresses
    localparam A_INPUT_BASE = 15'd0;
    localparam A_BOT_OUT    = 15'd4000;
    localparam A_CH1_OUT    = 15'd5000;
    localparam A_FINAL_OUT  = 15'd8000;
    localparam A_X_INNER    = 15'd12000;
    localparam A_Z_GATE     = 15'd20000;
    localparam A_MAMBA_OUT  = 15'd28128;  // was 20000; moved to preserve Z_GATE for debug
    localparam A_H_STATE    = 15'd28000;
    localparam B_P1_OUT     = 15'd0;
    localparam B_CH2_OUT    = 15'd4000;
    localparam B_CH3_OUT    = 15'd5000;
    localparam B_CH4_OUT    = 15'd6000;
    localparam B_FINAL_OUT  = 15'd8000;
    localparam B_X_CONV     = 15'd12000;
    localparam B_U_SAFE     = 15'd15000;  // was 20000; moved to reuse stale x_conv region
    localparam B_Y_SSM      = 15'd23000;  // was 24000; moved to avoid U_SAFE overlap
    // Dynamic weight base addresses - computed from block config
    localparam W_P1_BASE    = 15'd0;
    wire [14:0] w_p1_size   = {7'd0, CH_OUT} * {7'd0, d_in};             // CH_OUT * d_in
    wire [14:0] W_BOT_BASE  = W_P1_BASE + w_p1_size;
    wire [14:0] w_bot_size  = br_dim_groups * {7'd0, d_out};             // br_grps * d_out
    wire [14:0] W_B1_BASE   = W_BOT_BASE + w_bot_size;
    wire [14:0] w_b1_size   = w_bot_size;
    wire [14:0] W_B2_BASE   = W_B1_BASE + w_b1_size;
    wire [14:0] w_b2_size   = br_dim_groups * 15'd9 * {11'd0, CH_OUT, 2'b0};   // br_grps * 9 * dim
    wire [14:0] W_B3_BASE   = W_B2_BASE + w_b2_size;
    wire [14:0] w_b3_size   = br_dim_groups * 15'd19 * {11'd0, CH_OUT, 2'b0};
    wire [14:0] W_B4_BASE   = W_B3_BASE + w_b3_size;
    wire [14:0] w_b4_size   = br_dim_groups * 15'd39 * {11'd0, CH_OUT, 2'b0};
    wire [14:0] W_M_X_BASE  = W_B4_BASE + w_b4_size;
    wire [14:0] w_mx_size   = {7'd0, ch_m_actual} * {7'd0, d_out};      // CH_M * d_out
    wire [14:0] W_M_Z_BASE  = W_M_X_BASE + w_mx_size;
    wire [14:0] W_M_DW_BASE = W_M_Z_BASE + w_mx_size;                   // same size as M_X
    localparam C_P1_BIAS    = 15'd0;
    localparam C_INC_SCALE  = 15'd4;
    localparam C_INC_SHIFT  = 15'd8;
    localparam C_M_DW_BIAS  = 15'd12;
    localparam C_M_DT_BIAS  = 15'd20;

    // Memory interface
    reg          bank_sel;
    reg  [14:0]  m_rd_addr, m_wr_addr, w_rd_addr, c_rd_addr;
    wire [255:0] m_rd_data, w_rd_data, c_rd_data;
    reg          m_we;
    reg  [255:0] m_wr_data;
    reg          pe_clear;
    reg  [1:0]   pe_op_mode;
    reg  signed [15:0] pe_A;
    reg  [255:0] pe_A_vec;
    reg          pe_a_is_vector;
    reg  [255:0] pe_B;
    wire [255:0] pe_out;

    Memory_System mem_sys (
        .clk(clk), .reset(rst), .bank_sel(bank_sel),
        .core_read_addr(m_rd_addr),   .core_read_data(m_rd_data),
        .core_write_en(m_we),         .core_write_addr(m_wr_addr), .core_write_data(m_wr_data),
        .weight_read_addr(w_rd_addr), .weight_read_data(w_rd_data),
        .const_read_addr(c_rd_addr),  .const_read_data(c_rd_data),
        .dma_write_en(dma_write_en),  .dma_target(dma_target),
        .dma_addr(dma_addr),          .dma_wdata(dma_wdata)
    );

    PE_Array pe_arr (
        .clk(clk), .rst(rst),
        .clear_acc(pe_clear),   .op_mode(pe_op_mode),
        .in_A(pe_A),            .in_A_vec(pe_A_vec), .a_is_vector(pe_a_is_vector),
        .in_B(pe_B),
        .out_vector(pe_out)
    );

    // Activation LUTs (16 lanes each function)
    wire signed [15:0] silu_in [0:15], silu_o [0:15];
    wire signed [15:0] sp_in   [0:15], sp_o   [0:15];
    wire signed [15:0] exp_in  [0:15], exp_o  [0:15];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : ACT_LANES
            Activation_LUT lut_silu (.x_in(silu_in[gi]), .silu_out(silu_o[gi]), .softplus_out(),    .exp_out());
            Activation_LUT lut_sp   (.x_in(sp_in[gi]),   .silu_out(),           .softplus_out(sp_o[gi]),  .exp_out());
            Activation_LUT lut_exp  (.x_in(exp_in[gi]),  .silu_out(),           .softplus_out(),    .exp_out(exp_o[gi]));
        end
    endgenerate

    assign silu_in[ 0] = m_rd_data[  0 +: 16]; assign silu_in[ 1] = m_rd_data[ 16 +: 16];
    assign silu_in[ 2] = m_rd_data[ 32 +: 16]; assign silu_in[ 3] = m_rd_data[ 48 +: 16];
    assign silu_in[ 4] = m_rd_data[ 64 +: 16]; assign silu_in[ 5] = m_rd_data[ 80 +: 16];
    assign silu_in[ 6] = m_rd_data[ 96 +: 16]; assign silu_in[ 7] = m_rd_data[112 +: 16];
    assign silu_in[ 8] = m_rd_data[128 +: 16]; assign silu_in[ 9] = m_rd_data[144 +: 16];
    assign silu_in[10] = m_rd_data[160 +: 16]; assign silu_in[11] = m_rd_data[176 +: 16];
    assign silu_in[12] = m_rd_data[192 +: 16]; assign silu_in[13] = m_rd_data[208 +: 16];
    assign silu_in[14] = m_rd_data[224 +: 16]; assign silu_in[15] = m_rd_data[240 +: 16];

    reg signed [15:0] dt_lane [0:15];
    assign sp_in[ 0] = dt_lane[ 0]; assign sp_in[ 1] = dt_lane[ 1];
    assign sp_in[ 2] = dt_lane[ 2]; assign sp_in[ 3] = dt_lane[ 3];
    assign sp_in[ 4] = dt_lane[ 4]; assign sp_in[ 5] = dt_lane[ 5];
    assign sp_in[ 6] = dt_lane[ 6]; assign sp_in[ 7] = dt_lane[ 7];
    assign sp_in[ 8] = dt_lane[ 8]; assign sp_in[ 9] = dt_lane[ 9];
    assign sp_in[10] = dt_lane[10]; assign sp_in[11] = dt_lane[11];
    assign sp_in[12] = dt_lane[12]; assign sp_in[13] = dt_lane[13];
    assign sp_in[14] = dt_lane[14]; assign sp_in[15] = dt_lane[15];

    assign exp_in[ 0] = pe_out[  0 +: 16]; assign exp_in[ 1] = pe_out[ 16 +: 16];
    assign exp_in[ 2] = pe_out[ 32 +: 16]; assign exp_in[ 3] = pe_out[ 48 +: 16];
    assign exp_in[ 4] = pe_out[ 64 +: 16]; assign exp_in[ 5] = pe_out[ 80 +: 16];
    assign exp_in[ 6] = pe_out[ 96 +: 16]; assign exp_in[ 7] = pe_out[112 +: 16];
    assign exp_in[ 8] = pe_out[128 +: 16]; assign exp_in[ 9] = pe_out[144 +: 16];
    assign exp_in[10] = pe_out[160 +: 16]; assign exp_in[11] = pe_out[176 +: 16];
    assign exp_in[12] = pe_out[192 +: 16]; assign exp_in[13] = pe_out[208 +: 16];
    assign exp_in[14] = pe_out[224 +: 16]; assign exp_in[15] = pe_out[240 +: 16];

    // ================================================================
    // FSM state encoding
    // ================================================================
    localparam S_IDLE         = 7'd0;
    localparam S_P1_MAC       = 7'd1;   localparam S_P1_WAIT      = 7'd2;
    localparam S_P1_WRITE     = 7'd3;   localparam S_P1_NEXT      = 7'd4;
    localparam S_BR_MAC       = 7'd5;   localparam S_BR_WAIT      = 7'd6;
    localparam S_BR_WRITE     = 7'd7;   localparam S_BR_NEXT      = 7'd8;
    localparam S_M1A_MAC      = 7'd9;   localparam S_M1A_WAIT     = 7'd10;
    localparam S_M1A_WRITE    = 7'd11;  localparam S_M1A_NEXT     = 7'd12;
    localparam S_M1B_MAC      = 7'd13;  localparam S_M1B_WAIT     = 7'd14;
    localparam S_M1B_WRITE    = 7'd15;  localparam S_M1B_NEXT     = 7'd16;
    localparam S_M2_MAC       = 7'd17;  localparam S_M2_WAIT      = 7'd18;
    localparam S_M2_WRITE     = 7'd19;  localparam S_M2_NEXT      = 7'd20;
    localparam S_M3_READ      = 7'd21;  localparam S_M3_WAIT      = 7'd22;
    localparam S_M3_WRITE     = 7'd23;  localparam S_M3_NEXT      = 7'd24;
    localparam S_M4_MAC       = 7'd25;  localparam S_M4_WAIT      = 7'd26;
    localparam S_M4_WRITE     = 7'd27;  localparam S_M4_NEXT      = 7'd28;
    localparam S_M5_MAC       = 7'd29;  localparam S_M5_WAIT      = 7'd30;
    localparam S_M5_LATCH     = 7'd39;  localparam S_M5_WRITE     = 7'd31;
    localparam S_M5_NEXT      = 7'd32;
    localparam S_M3CP_READ    = 7'd40;  localparam S_M3CP_WAIT    = 7'd41;
    localparam S_M3CP_LATCH   = 7'd42;  localparam S_M3CP_WRITE   = 7'd43;
    localparam S_M3CP_NEXT    = 7'd44;
    localparam S_M6A_INIT_H   = 7'd45;  localparam S_M6A_INIT_NEXT = 7'd46;
    localparam S_M6A_DA_READ  = 7'd47;  localparam S_M6A_DA_WAIT  = 7'd48;
    localparam S_M6A_DA_LATCH = 7'd49;  localparam S_M6A_DA_WAIT2 = 7'd50;
    localparam S_M6A_DA_CAP   = 7'd51;
    localparam S_M6A_DB_READ  = 7'd52;  localparam S_M6A_DB_WAIT  = 7'd53;
    localparam S_M6A_DB_LATCH = 7'd54;  localparam S_M6A_DB_WAIT2 = 7'd55;
    localparam S_M6A_DB_CAP   = 7'd56;
    localparam S_M6A_T1_READ  = 7'd57;  localparam S_M6A_T1_WAIT  = 7'd58;
    localparam S_M6A_T1_LATCH = 7'd59;  localparam S_M6A_T1_WAIT2 = 7'd60;
    localparam S_M6A_T1_CAP   = 7'd61;
    localparam S_M6A_T2_READ  = 7'd62;  localparam S_M6A_T2_WAIT  = 7'd64;
    localparam S_M6A_T2_LATCH = 7'd65;  localparam S_M6A_T2_WAIT2 = 7'd66;
    localparam S_M6A_T2_CAP   = 7'd67;
    localparam S_M6A_HW       = 7'd68;  localparam S_M6A_NEXT     = 7'd69;
    localparam S_M6B_INIT     = 7'd70;  localparam S_M6B_RH_READ  = 7'd71;
    localparam S_M6B_RH_WAIT  = 7'd72;  localparam S_M6B_RH_LATCH = 7'd73;
    localparam S_M6B_RC_READ  = 7'd74;  localparam S_M6B_RC_WAIT  = 7'd75;
    localparam S_M6B_RC_LATCH = 7'd76;  localparam S_M6B_RC_WAIT2 = 7'd77;
    localparam S_M6B_S_NEXT   = 7'd78;  localparam S_M6B_CAP_Y    = 7'd79;
    localparam S_M6B_DU_READ  = 7'd80;  localparam S_M6B_DU_WAIT  = 7'd81;
    localparam S_M6B_DU_LATCH = 7'd82;  localparam S_M6B_DU_WAIT2 = 7'd83;
    localparam S_M6B_DU_CAP   = 7'd84;  localparam S_M6B_WRITE    = 7'd85;
    localparam S_M6B_NEXT     = 7'd86;
    localparam S_M7_RY_READ   = 7'd100; localparam S_M7_RY_WAIT   = 7'd101;
    localparam S_M7_RY_LATCH  = 7'd102; localparam S_M7_RZ_READ   = 7'd103;
    localparam S_M7_RZ_WAIT   = 7'd104; localparam S_M7_RZ_LATCH  = 7'd105;
    localparam S_M7_PE_WAIT2  = 7'd106; localparam S_M7_WRITE     = 7'd107;
    localparam S_M7_NEXT      = 7'd108;
    localparam S_M8_MAC       = 7'd109; localparam S_M8_WAIT      = 7'd110;
    localparam S_M8_WRITE     = 7'd111; localparam S_M8_NEXT      = 7'd112;
    localparam S_FIN_READ     = 7'd33;  localparam S_FIN_WAIT     = 7'd34;
    localparam S_FIN_MUL      = 7'd35;  localparam S_FIN_WAIT2    = 7'd36;
    localparam S_FIN_READ_M   = 7'd113; localparam S_FIN_WAIT_M   = 7'd114;
    localparam S_FIN_WRITE    = 7'd37;  localparam S_FIN_NEXT     = 7'd38;
    localparam S_DONE         = 7'd63;

    // ================================================================
    // FSM registers
    // ================================================================
    reg [6:0]  state;
    reg [9:0]  t_cnt;
    reg [3:0]  c_grp_m;
    reg [2:0]  c_grp;
    reg [7:0]  mac_idx;
    reg [5:0]  k_idx;
    reg [1:0]  substep;
    reg [2:0]  branch_id;
    reg [3:0]  s_idx;
    reg        c_grp_br;   // inception output-group within branch (0 or 1)

    reg signed [15:0] dA_reg    [0:15];
    reg signed [15:0] dB_reg    [0:15];
    reg signed [15:0] term1_reg [0:15];
    reg signed [15:0] term2_reg [0:15];
    reg signed [15:0] B_scalar_reg, C_scalar_reg;
    reg signed [15:0] u_scalar_reg [0:15];
    reg        [255:0] h_reg;
    reg signed [15:0] y_acc_reg [0:15];
    reg signed [15:0] du_reg    [0:15];
    reg        [255:0] y_reg;
    reg        [255:0] incep_reg;

    // ================================================================
    // Inception helpers - block-4 aware
    // ================================================================
    wire is_ch64_branch = (branch_id == 3'd0 || branch_id == 3'd1);

    // dim = d_out/4 = CH_IN*4  (16 for blk 0-3, 32 for blk 4)
    // Bot/B1: input = P1_out = d_out channels = CH_IN*16
    // B2/B3/B4: input = Bot_out = dim channels = CH_IN*4
    wire [7:0] current_num_in_ch = is_ch64_branch ? {CH_OUT, 4'b0}   // Bot/B1 input = d_out channels
                                                   : {2'b0, CH_OUT, 2'b0}; // B2/B3/B4 input = dim = d_out/4

    wire [5:0] current_kernel =
        (branch_id == 3'd0 || branch_id == 3'd1) ? 6'd1  :
        (branch_id == 3'd2) ? 6'd9  :
        (branch_id == 3'd3) ? 6'd19 : 6'd39;

    wire [5:0] current_pad =
        (branch_id == 3'd0 || branch_id == 3'd1) ? 6'd0 :
        (branch_id == 3'd2) ? 6'd4  :
        (branch_id == 3'd3) ? 6'd9  : 6'd19;

    wire [14:0] current_w_base =
        (branch_id == 3'd0) ? W_BOT_BASE :
        (branch_id == 3'd1) ? W_B1_BASE  :
        (branch_id == 3'd2) ? W_B2_BASE  :
        (branch_id == 3'd3) ? W_B3_BASE  : W_B4_BASE;

    wire [14:0] current_data_base =
        (branch_id == 3'd0 || branch_id == 3'd1) ? B_P1_OUT : A_BOT_OUT;

    wire [14:0] current_out_base =
        (branch_id == 3'd0) ? A_BOT_OUT :
        (branch_id == 3'd1) ? A_CH1_OUT :
        (branch_id == 3'd2) ? B_CH2_OUT :
        (branch_id == 3'd3) ? B_CH3_OUT : B_CH4_OUT;

    // br_grp_last: 0 for blk 0-3, 1 for blk 4
    wire        br_grp_last  = (CH_OUT >= 4'd8) ? 1'b1 : 1'b0;
    // words per branch per timestep: 1 or 2
    wire [14:0] br_dim_groups = {14'd0, br_grp_last} + 15'd1;

    // Weight address offset for c_grp_br
    wire [14:0] d_out_15     = {7'd0, d_out};  // d_out = CH_OUT*16
    wire [14:0] br_w_offset  = is_ch64_branch
        ? ({14'd0, c_grp_br} * d_out_15)
        : ({14'd0, c_grp_br} * ({9'd0, current_kernel} * {7'd0, current_num_in_ch}));

    // Inception t_eff
    wire signed [11:0] t_eff_signed =
        {{2{1'b0}}, t_cnt} + {{6{1'b0}}, k_idx} - {{6{1'b0}}, current_pad};
    wire        is_padding  = (t_eff_signed < 12'sd0) || (t_eff_signed >= {2'b0, T_MAX});
    wire [9:0]  t_eff       = t_eff_signed[9:0];
    wire [7:0]  mac_target  = current_num_in_ch - 8'd1;
    wire [5:0]  k_target    = current_kernel - 6'd1;

    // M2 causal pad=3
    wire signed [11:0] m2_t_eff_signed = {{2{1'b0}}, t_cnt} + {{6{1'b0}}, k_idx} - 12'sd3;
    wire        m2_is_padding = (m2_t_eff_signed < 12'sd0);
    wire [9:0]  m2_t_eff    = m2_t_eff_signed[9:0];

    // Final read - generalised to 8 output groups (block 4)
    // fin_branch = c_grp / br_dim_groups = c_grp >> br_grp_last
    // blocks 0-3 (br_grp_last=0): fin_branch = c_grp[1:0], fin_sub = 0
    // block 4    (br_grp_last=1): fin_branch = c_grp[2:1], fin_sub = c_grp[0]
    wire [1:0]  fin_branch      = br_grp_last ? c_grp[2:1] : c_grp[1:0];
    wire        fin_sub         = c_grp[0] & br_grp_last;
    wire [14:0] fin_branch_base =
        (fin_branch == 2'd0) ? A_CH1_OUT :
        (fin_branch == 2'd1) ? B_CH2_OUT :
        (fin_branch == 2'd2) ? B_CH3_OUT : B_CH4_OUT;
    wire        fin_branch_bank = (fin_branch == 2'd0) ? 1'b0 : 1'b1;

    // B/C xproj lane selectors
    wire [4:0]  b_lane_idx   = {1'b0, DT_RANK[3:0]} + {1'b0, s_idx};
    wire [3:0]  b_lane       = b_lane_idx[3:0];
    wire [14:0] b_grp_offset = {14'd0, b_lane_idx[4]};
    wire [5:0]  c_lane_idx   = {2'b0, DT_RANK[3:0]} + 6'd16 + {2'b0, s_idx};
    wire [3:0]  c_lane       = c_lane_idx[3:0];
    wire [14:0] c_grp_offset = {13'd0, c_lane_idx[5:4]};

    integer i;

    // ================================================================
    // FSM
    // ================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            t_cnt <= 0; c_grp <= 0; c_grp_m <= 0; mac_idx <= 0;
            k_idx <= 0; substep <= 0; branch_id <= 0; s_idx <= 0; c_grp_br <= 0;
            done_phase1 <= 0; done_inception <= 0; done_mamba <= 0; done_all <= 0;
            bank_sel <= 0; m_we <= 0;
            m_rd_addr <= 0; m_wr_addr <= 0; w_rd_addr <= 0; c_rd_addr <= 0;
            pe_clear <= 0; pe_A <= 0; pe_A_vec <= 0; pe_a_is_vector <= 0;
            pe_B <= 0; m_wr_data <= 0; pe_op_mode <= `MODE_MAC;
            for (i = 0; i < 16; i = i + 1) dt_lane[i] <= 0;
            B_scalar_reg <= 0; C_scalar_reg <= 0;
            for (i = 0; i < 16; i = i + 1) begin
                dA_reg[i] <= 0; dB_reg[i] <= 0;
                term1_reg[i] <= 0; term2_reg[i] <= 0;
                u_scalar_reg[i] <= 0; y_acc_reg[i] <= 0; du_reg[i] <= 0;
            end
            h_reg <= 0; y_reg <= 0; incep_reg <= 0;
        end else begin
            m_we     <= 0;
            pe_clear <= 0;
            case (state)

            // --------------------------------------------------------
            // IDLE
            // --------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    done_phase1 <= 0; done_inception <= 0;
                    done_mamba  <= 0; done_all       <= 0;
                    t_cnt <= 0; c_grp <= 0; c_grp_m <= 0; mac_idx <= 0;
                    k_idx <= 0; substep <= 0; branch_id <= 0; s_idx <= 0; c_grp_br <= 0;
                    bank_sel  <= 0;
                    c_rd_addr <= C_P1_BIAS;
                    state     <= S_P1_MAC;
                end
            end

            // --------------------------------------------------------
            // PHASE 1 : Conv1D + BN  t x c_grp x mac_idx
            // --------------------------------------------------------
            S_P1_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= A_INPUT_BASE + t_stride_in + {10'd0, mac_idx[7:4]};
                        w_rd_addr <= W_P1_BASE + ({7'd0, c_grp} * {7'd0, d_in}) + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        pe_A     <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == d_in_last) state <= S_P1_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_P1_WAIT:  state <= S_P1_WRITE;
            S_P1_WRITE: begin
                m_we      <= 1;
                m_wr_addr <= B_P1_OUT + t_stride_out + {12'd0, c_grp};
                m_wr_data[  0 +: 16] <= sat_add16(pe_out[  0 +: 16], c_rd_data[  0 +: 16]);
                m_wr_data[ 16 +: 16] <= sat_add16(pe_out[ 16 +: 16], c_rd_data[ 16 +: 16]);
                m_wr_data[ 32 +: 16] <= sat_add16(pe_out[ 32 +: 16], c_rd_data[ 32 +: 16]);
                m_wr_data[ 48 +: 16] <= sat_add16(pe_out[ 48 +: 16], c_rd_data[ 48 +: 16]);
                m_wr_data[ 64 +: 16] <= sat_add16(pe_out[ 64 +: 16], c_rd_data[ 64 +: 16]);
                m_wr_data[ 80 +: 16] <= sat_add16(pe_out[ 80 +: 16], c_rd_data[ 80 +: 16]);
                m_wr_data[ 96 +: 16] <= sat_add16(pe_out[ 96 +: 16], c_rd_data[ 96 +: 16]);
                m_wr_data[112 +: 16] <= sat_add16(pe_out[112 +: 16], c_rd_data[112 +: 16]);
                m_wr_data[128 +: 16] <= sat_add16(pe_out[128 +: 16], c_rd_data[128 +: 16]);
                m_wr_data[144 +: 16] <= sat_add16(pe_out[144 +: 16], c_rd_data[144 +: 16]);
                m_wr_data[160 +: 16] <= sat_add16(pe_out[160 +: 16], c_rd_data[160 +: 16]);
                m_wr_data[176 +: 16] <= sat_add16(pe_out[176 +: 16], c_rd_data[176 +: 16]);
                m_wr_data[192 +: 16] <= sat_add16(pe_out[192 +: 16], c_rd_data[192 +: 16]);
                m_wr_data[208 +: 16] <= sat_add16(pe_out[208 +: 16], c_rd_data[208 +: 16]);
                m_wr_data[224 +: 16] <= sat_add16(pe_out[224 +: 16], c_rd_data[224 +: 16]);
                m_wr_data[240 +: 16] <= sat_add16(pe_out[240 +: 16], c_rd_data[240 +: 16]);
                state <= S_P1_NEXT;
            end
            S_P1_NEXT: begin
                mac_idx <= 0; substep <= 0;
                if (c_grp == ch_out_last[2:0]) begin
                    c_grp <= 0;
                    if (t_cnt == t_last) begin
                        done_phase1 <= 1; t_cnt <= 0; k_idx <= 0;
                        branch_id <= 0; bank_sel <= 1; c_grp_br <= 0;
                        state <= S_BR_MAC;
                    end else begin
                        t_cnt     <= t_cnt + 10'd1;
                        c_rd_addr <= C_P1_BIAS;
                        state     <= S_P1_MAC;
                    end
                end else begin
                    c_grp     <= c_grp + 3'd1;
                    c_rd_addr <= C_P1_BIAS + {12'd0, c_grp} + 15'd1;
                    state     <= S_P1_MAC;
                end
            end

            // --------------------------------------------------------
            // PHASE 2 : Inception  branch_id x c_grp_br x t x k x mac
            // --------------------------------------------------------
            S_BR_MAC: begin
                case (substep)
                    2'd0: begin
                        if (is_padding)
                            m_rd_addr <= 15'd0;
                        else if (is_ch64_branch)
                            // Bot/B1: d_out channels, CH_OUT words per t
                            m_rd_addr <= current_data_base
                                       + ({5'd0, t_eff} * {11'd0, CH_OUT})
                                       + {10'd0, mac_idx[7:4]};
                        else
                            // B2/B3/B4: dim channels, br_dim_groups words per t
                            // mac_idx[4] selects sub-word for block 4 (dim=32)
                            m_rd_addr <= current_data_base
                                       + ({5'd0, t_eff} * br_dim_groups)
                                       + {14'd0, mac_idx[4]};
                        // Weight: base + output-group-offset + tap*num_in + mac
                        w_rd_addr <= current_w_base + br_w_offset
                                   + ({9'd0, k_idx} * {7'd0, current_num_in_ch})
                                   + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        if (is_padding) pe_A <= 16'sd0;
                        else            pe_A <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (k_idx == 6'd0 && mac_idx == 8'd0);
                        if (mac_idx == mac_target[7:0]) begin
                            mac_idx <= 0;
                            if (k_idx == k_target) state <= S_BR_WAIT;
                            else begin k_idx <= k_idx + 6'd1; substep <= 2'd0; end
                        end else begin
                            mac_idx <= mac_idx + 8'd1; substep <= 2'd0;
                        end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_BR_WAIT:  state <= S_BR_WRITE;
            S_BR_WRITE: begin
                m_we      <= 1;
                m_wr_addr <= current_out_base
                           + ({5'd0, t_cnt} * br_dim_groups)
                           + {14'd0, c_grp_br};
                m_wr_data <= pe_out;
                state     <= S_BR_NEXT;
            end
            S_BR_NEXT: begin
                mac_idx <= 0; k_idx <= 0; substep <= 0;
                if (t_cnt == t_last) begin
                    t_cnt <= 0;
                    if (c_grp_br < br_grp_last) begin
                        // More output groups in this branch
                        c_grp_br <= c_grp_br + 1'b1;
                        state    <= S_BR_MAC;
                    end else begin
                        c_grp_br <= 0;
                        case (branch_id)
                            3'd0: begin branch_id <= 3'd1; bank_sel <= 1; state <= S_BR_MAC; end
                            3'd1: begin branch_id <= 3'd2; bank_sel <= 0; state <= S_BR_MAC; end
                            3'd2: begin branch_id <= 3'd3; bank_sel <= 0; state <= S_BR_MAC; end
                            3'd3: begin branch_id <= 3'd4; bank_sel <= 0; state <= S_BR_MAC; end
                            3'd4: begin
                                done_inception <= 1;
                                t_cnt <= 0; c_grp_m <= 0; mac_idx <= 0;
                                bank_sel <= 1;
                                state    <= S_M1A_MAC;
                            end
                            default: state <= S_DONE;
                        endcase
                    end
                end else begin
                    t_cnt <= t_cnt + 10'd1;
                    state <= S_BR_MAC;
                end
            end

            // --------------------------------------------------------
            // M1a : in_proj_x   d_out -> d_inner
            // --------------------------------------------------------
            S_M1A_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= B_P1_OUT + t_stride_out + {10'd0, mac_idx[7:4]};
                        w_rd_addr <= W_M_X_BASE + ({7'd0, c_grp_m} * {7'd0, d_out}) + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        pe_A     <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == d_out - 8'd1) state <= S_M1A_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_M1A_WAIT:  state <= S_M1A_WRITE;
            S_M1A_WRITE: begin
                m_we <= 1; m_wr_addr <= A_X_INNER + t_stride_m + {11'd0, c_grp_m};
                m_wr_data <= pe_out; state <= S_M1A_NEXT;
            end
            S_M1A_NEXT: begin
                mac_idx <= 0; substep <= 0;
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; state <= S_M1B_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M1A_MAC; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M1A_MAC; end
            end

            // --------------------------------------------------------
            // M1b : in_proj_z   d_out -> d_inner  -> A_Z_GATE
            // --------------------------------------------------------
            S_M1B_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= B_P1_OUT + t_stride_out + {10'd0, mac_idx[7:4]};
                        w_rd_addr <= W_M_Z_BASE + ({7'd0, c_grp_m} * {7'd0, d_out}) + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        pe_A     <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == d_out - 8'd1) state <= S_M1B_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_M1B_WAIT:  state <= S_M1B_WRITE;
            S_M1B_WRITE: begin
                m_we <= 1; m_wr_addr <= A_Z_GATE + t_stride_m + {11'd0, c_grp_m};
                m_wr_data <= pe_out; state <= S_M1B_NEXT;
            end
            S_M1B_NEXT: begin
                mac_idx <= 0; substep <= 0;
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; k_idx <= 0; bank_sel <= 0; state <= S_M2_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M1B_MAC; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M1B_MAC; end
            end

            // --------------------------------------------------------
            // M2 : depthwise conv1d  k=4  causal pad=3
            // --------------------------------------------------------
            S_M2_MAC: begin
                case (substep)
                    2'd0: begin
                        if (m2_is_padding) m_rd_addr <= 15'd0;
                        else m_rd_addr <= A_X_INNER + ({5'd0, m2_t_eff} * {7'd0, ch_m_actual}) + {11'd0, c_grp_m};
                        w_rd_addr <= W_M_DW_BASE + ({10'd0, c_grp_m} * 15'd4) + {9'd0, k_idx};
                        if (k_idx == 6'd0) c_rd_addr <= C_M_DW_BIAS + {11'd0, c_grp_m};
                        pe_A_vec <= 256'd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A_vec <= 256'd0; substep <= 2'd2; end
                    2'd2: begin
                        if (m2_is_padding) pe_A_vec <= 256'd0;
                        else               pe_A_vec <= m_rd_data;
                        pe_B           <= w_rd_data;
                        pe_a_is_vector <= 1'b1;
                        pe_op_mode     <= `MODE_MAC;
                        pe_clear       <= (k_idx == 6'd0);
                        if (k_idx == 6'd3) state <= S_M2_WAIT;
                        else begin k_idx <= k_idx + 6'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_M2_WAIT: begin pe_a_is_vector <= 1'b0; state <= S_M2_WRITE; end
            S_M2_WRITE: begin
                m_we      <= 1;
                m_wr_addr <= B_X_CONV + t_stride_m + {11'd0, c_grp_m};
                m_wr_data[  0 +: 16] <= sat_add16(pe_out[  0 +: 16], c_rd_data[  0 +: 16]);
                m_wr_data[ 16 +: 16] <= sat_add16(pe_out[ 16 +: 16], c_rd_data[ 16 +: 16]);
                m_wr_data[ 32 +: 16] <= sat_add16(pe_out[ 32 +: 16], c_rd_data[ 32 +: 16]);
                m_wr_data[ 48 +: 16] <= sat_add16(pe_out[ 48 +: 16], c_rd_data[ 48 +: 16]);
                m_wr_data[ 64 +: 16] <= sat_add16(pe_out[ 64 +: 16], c_rd_data[ 64 +: 16]);
                m_wr_data[ 80 +: 16] <= sat_add16(pe_out[ 80 +: 16], c_rd_data[ 80 +: 16]);
                m_wr_data[ 96 +: 16] <= sat_add16(pe_out[ 96 +: 16], c_rd_data[ 96 +: 16]);
                m_wr_data[112 +: 16] <= sat_add16(pe_out[112 +: 16], c_rd_data[112 +: 16]);
                m_wr_data[128 +: 16] <= sat_add16(pe_out[128 +: 16], c_rd_data[128 +: 16]);
                m_wr_data[144 +: 16] <= sat_add16(pe_out[144 +: 16], c_rd_data[144 +: 16]);
                m_wr_data[160 +: 16] <= sat_add16(pe_out[160 +: 16], c_rd_data[160 +: 16]);
                m_wr_data[176 +: 16] <= sat_add16(pe_out[176 +: 16], c_rd_data[176 +: 16]);
                m_wr_data[192 +: 16] <= sat_add16(pe_out[192 +: 16], c_rd_data[192 +: 16]);
                m_wr_data[208 +: 16] <= sat_add16(pe_out[208 +: 16], c_rd_data[208 +: 16]);
                m_wr_data[224 +: 16] <= sat_add16(pe_out[224 +: 16], c_rd_data[224 +: 16]);
                m_wr_data[240 +: 16] <= sat_add16(pe_out[240 +: 16], c_rd_data[240 +: 16]);
                state <= S_M2_NEXT;
            end
            S_M2_NEXT: begin
                k_idx <= 0; substep <= 0;
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; bank_sel <= 1; state <= S_M3_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M2_MAC; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M2_MAC; end
            end

            // --------------------------------------------------------
            // M3 : SiLU via LUT
            // --------------------------------------------------------
            S_M3_READ: begin
                m_rd_addr <= B_X_CONV + t_stride_m + {11'd0, c_grp_m}; state <= S_M3_WAIT;
            end
            S_M3_WAIT: state <= S_M3_WRITE;
            S_M3_WRITE: begin
                m_we      <= 1;
                m_wr_addr <= A_X_INNER + t_stride_m + {11'd0, c_grp_m};
                m_wr_data[  0 +: 16] <= silu_o[ 0]; m_wr_data[ 16 +: 16] <= silu_o[ 1];
                m_wr_data[ 32 +: 16] <= silu_o[ 2]; m_wr_data[ 48 +: 16] <= silu_o[ 3];
                m_wr_data[ 64 +: 16] <= silu_o[ 4]; m_wr_data[ 80 +: 16] <= silu_o[ 5];
                m_wr_data[ 96 +: 16] <= silu_o[ 6]; m_wr_data[112 +: 16] <= silu_o[ 7];
                m_wr_data[128 +: 16] <= silu_o[ 8]; m_wr_data[144 +: 16] <= silu_o[ 9];
                m_wr_data[160 +: 16] <= silu_o[10]; m_wr_data[176 +: 16] <= silu_o[11];
                m_wr_data[192 +: 16] <= silu_o[12]; m_wr_data[208 +: 16] <= silu_o[13];
                m_wr_data[224 +: 16] <= silu_o[14]; m_wr_data[240 +: 16] <= silu_o[15];
                state <= S_M3_NEXT;
            end
            S_M3_NEXT: begin
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; substep <= 0; bank_sel <= 0; state <= S_M3CP_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M3_READ; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M3_READ; end
            end

            // --------------------------------------------------------
            // M3_COPY : u_silu A_X_INNER -> B_U_SAFE
            // --------------------------------------------------------
            S_M3CP_READ: begin
                m_rd_addr <= A_X_INNER + t_stride_m + {11'd0, c_grp_m}; state <= S_M3CP_WAIT;
            end
            S_M3CP_WAIT:  state <= S_M3CP_LATCH;
            S_M3CP_LATCH: begin m_wr_data <= m_rd_data; state <= S_M3CP_WRITE; end
            S_M3CP_WRITE: begin
                m_we <= 1; m_wr_addr <= B_U_SAFE + t_stride_m + {11'd0, c_grp_m}; state <= S_M3CP_NEXT;
            end
            S_M3CP_NEXT: begin
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp <= 0; mac_idx <= 0; substep <= 0; bank_sel <= 0; state <= S_M4_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M3CP_READ; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M3CP_READ; end
            end

            // --------------------------------------------------------
            // M4 : x_proj  d_inner -> 48  (3 groups fixed)
            // --------------------------------------------------------
            S_M4_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= A_X_INNER + t_stride_m + {7'd0, mac_idx[7:4]};
                        w_rd_addr <= W_XPROJ_BASE_W + ({12'd0, c_grp} * {7'd0, d_inner}) + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        pe_A     <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == d_inner_last) state <= S_M4_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_M4_WAIT:  state <= S_M4_WRITE;
            S_M4_WRITE: begin
                m_we <= 1; m_wr_addr <= B_X_CONV + t_stride_xp + {12'd0, c_grp};
                m_wr_data <= pe_out; state <= S_M4_NEXT;
            end
            S_M4_NEXT: begin
                mac_idx <= 0; substep <= 0;
                if (c_grp == 3'd2) begin
                    c_grp <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; mac_idx <= 0; k_idx <= 0; substep <= 0; bank_sel <= 1; state <= S_M5_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M4_MAC; end
                end else begin c_grp <= c_grp + 3'd1; state <= S_M4_MAC; end
            end

            // --------------------------------------------------------
            // M5 : dt_proj  DT_RANK -> d_inner + bias + softplus
            // --------------------------------------------------------
            S_M5_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= B_X_CONV + t_stride_xp + 15'd0;
                        w_rd_addr <= W_DTPROJ_BASE + ({11'd0, c_grp_m} * {11'd0, DT_RANK}) + {4'd0, mac_idx[3:0]};
                        if (mac_idx == 8'd0) c_rd_addr <= C_M_DT_BIAS + {11'd0, c_grp_m};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        pe_A     <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == {4'd0, dt_last}) state <= S_M5_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_M5_WAIT: state <= S_M5_LATCH;
            S_M5_LATCH: begin
                dt_lane[ 0] <= sat_add16(pe_out[  0 +: 16], c_rd_data[  0 +: 16]);
                dt_lane[ 1] <= sat_add16(pe_out[ 16 +: 16], c_rd_data[ 16 +: 16]);
                dt_lane[ 2] <= sat_add16(pe_out[ 32 +: 16], c_rd_data[ 32 +: 16]);
                dt_lane[ 3] <= sat_add16(pe_out[ 48 +: 16], c_rd_data[ 48 +: 16]);
                dt_lane[ 4] <= sat_add16(pe_out[ 64 +: 16], c_rd_data[ 64 +: 16]);
                dt_lane[ 5] <= sat_add16(pe_out[ 80 +: 16], c_rd_data[ 80 +: 16]);
                dt_lane[ 6] <= sat_add16(pe_out[ 96 +: 16], c_rd_data[ 96 +: 16]);
                dt_lane[ 7] <= sat_add16(pe_out[112 +: 16], c_rd_data[112 +: 16]);
                dt_lane[ 8] <= sat_add16(pe_out[128 +: 16], c_rd_data[128 +: 16]);
                dt_lane[ 9] <= sat_add16(pe_out[144 +: 16], c_rd_data[144 +: 16]);
                dt_lane[10] <= sat_add16(pe_out[160 +: 16], c_rd_data[160 +: 16]);
                dt_lane[11] <= sat_add16(pe_out[176 +: 16], c_rd_data[176 +: 16]);
                dt_lane[12] <= sat_add16(pe_out[192 +: 16], c_rd_data[192 +: 16]);
                dt_lane[13] <= sat_add16(pe_out[208 +: 16], c_rd_data[208 +: 16]);
                dt_lane[14] <= sat_add16(pe_out[224 +: 16], c_rd_data[224 +: 16]);
                dt_lane[15] <= sat_add16(pe_out[240 +: 16], c_rd_data[240 +: 16]);
                state <= S_M5_WRITE;
            end
            S_M5_WRITE: begin
                m_we      <= 1;
                m_wr_addr <= A_X_INNER + t_stride_m + {11'd0, c_grp_m};
                m_wr_data[  0 +: 16] <= sp_o[ 0]; m_wr_data[ 16 +: 16] <= sp_o[ 1];
                m_wr_data[ 32 +: 16] <= sp_o[ 2]; m_wr_data[ 48 +: 16] <= sp_o[ 3];
                m_wr_data[ 64 +: 16] <= sp_o[ 4]; m_wr_data[ 80 +: 16] <= sp_o[ 5];
                m_wr_data[ 96 +: 16] <= sp_o[ 6]; m_wr_data[112 +: 16] <= sp_o[ 7];
                m_wr_data[128 +: 16] <= sp_o[ 8]; m_wr_data[144 +: 16] <= sp_o[ 9];
                m_wr_data[160 +: 16] <= sp_o[10]; m_wr_data[176 +: 16] <= sp_o[11];
                m_wr_data[192 +: 16] <= sp_o[12]; m_wr_data[208 +: 16] <= sp_o[13];
                m_wr_data[224 +: 16] <= sp_o[14]; m_wr_data[240 +: 16] <= sp_o[15];
                state <= S_M5_NEXT;
            end
            S_M5_NEXT: begin
                mac_idx <= 0; substep <= 0;
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; s_idx <= 0; bank_sel <= 1; state <= S_M6A_INIT_H;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M5_MAC; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M5_MAC; end
            end

            // --------------------------------------------------------
            // M6a : h = dA*h + dB*u
            // --------------------------------------------------------
            S_M6A_INIT_H: begin
                m_we      <= 1; bank_sel <= 1;
                m_wr_addr <= A_H_STATE + ({11'd0, s_idx} * {7'd0, ch_m_actual}) + {11'd0, c_grp_m};
                m_wr_data <= 256'd0; state <= S_M6A_INIT_NEXT;
            end
            S_M6A_INIT_NEXT: begin
                if (s_idx == 4'd15) begin
                    s_idx <= 0;
                    if (c_grp_m == ch_m_last) begin
                        t_cnt <= 0; c_grp_m <= 0; s_idx <= 0; bank_sel <= 1; state <= S_M6A_DA_READ;
                    end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M6A_INIT_H; end
                end else begin s_idx <= s_idx + 4'd1; state <= S_M6A_INIT_H; end
            end

            S_M6A_DA_READ: begin
                bank_sel  <= 0;
                m_rd_addr <= A_X_INNER + t_stride_m + {11'd0, c_grp_m};
                w_rd_addr <= W_ALOG_BASE + ({11'd0, c_grp_m} * 15'd16) + {11'd0, s_idx};
                state     <= S_M6A_DA_WAIT;
            end
            S_M6A_DA_WAIT:  state <= S_M6A_DA_LATCH;
            S_M6A_DA_LATCH: begin
                pe_A_vec <= m_rd_data; pe_B <= w_rd_data;
                pe_a_is_vector <= 1; pe_op_mode <= `MODE_MUL; pe_clear <= 0;
                state <= S_M6A_DA_WAIT2;
            end
            S_M6A_DA_WAIT2: state <= S_M6A_DA_CAP;
            S_M6A_DA_CAP: begin
                dA_reg[ 0] <= exp_o[ 0]; dA_reg[ 1] <= exp_o[ 1];
                dA_reg[ 2] <= exp_o[ 2]; dA_reg[ 3] <= exp_o[ 3];
                dA_reg[ 4] <= exp_o[ 4]; dA_reg[ 5] <= exp_o[ 5];
                dA_reg[ 6] <= exp_o[ 6]; dA_reg[ 7] <= exp_o[ 7];
                dA_reg[ 8] <= exp_o[ 8]; dA_reg[ 9] <= exp_o[ 9];
                dA_reg[10] <= exp_o[10]; dA_reg[11] <= exp_o[11];
                dA_reg[12] <= exp_o[12]; dA_reg[13] <= exp_o[13];
                dA_reg[14] <= exp_o[14]; dA_reg[15] <= exp_o[15];
                state <= S_M6A_DB_READ;
            end

            S_M6A_DB_READ: begin
                bank_sel  <= 1;
                m_rd_addr <= B_X_CONV + t_stride_xp + b_grp_offset;
                state     <= S_M6A_DB_WAIT;
            end
            S_M6A_DB_WAIT:  state <= S_M6A_DB_LATCH;
            S_M6A_DB_LATCH: begin
                pe_B           <= {16{m_rd_data[b_lane * 16 +: 16]}};
                B_scalar_reg   <= m_rd_data[b_lane * 16 +: 16];
                pe_a_is_vector <= 1; pe_op_mode <= `MODE_MUL;
                state          <= S_M6A_DB_WAIT2;
            end
            S_M6A_DB_WAIT2: state <= S_M6A_DB_CAP;
            S_M6A_DB_CAP: begin
                dB_reg[ 0] <= pe_out[  0 +: 16]; dB_reg[ 1] <= pe_out[ 16 +: 16];
                dB_reg[ 2] <= pe_out[ 32 +: 16]; dB_reg[ 3] <= pe_out[ 48 +: 16];
                dB_reg[ 4] <= pe_out[ 64 +: 16]; dB_reg[ 5] <= pe_out[ 80 +: 16];
                dB_reg[ 6] <= pe_out[ 96 +: 16]; dB_reg[ 7] <= pe_out[112 +: 16];
                dB_reg[ 8] <= pe_out[128 +: 16]; dB_reg[ 9] <= pe_out[144 +: 16];
                dB_reg[10] <= pe_out[160 +: 16]; dB_reg[11] <= pe_out[176 +: 16];
                dB_reg[12] <= pe_out[192 +: 16]; dB_reg[13] <= pe_out[208 +: 16];
                dB_reg[14] <= pe_out[224 +: 16]; dB_reg[15] <= pe_out[240 +: 16];
                state <= S_M6A_T1_READ;
            end

            S_M6A_T1_READ: begin
                bank_sel  <= 0;
                m_rd_addr <= A_H_STATE + ({11'd0, s_idx} * {7'd0, ch_m_actual}) + {11'd0, c_grp_m};
                state     <= S_M6A_T1_WAIT;
            end
            S_M6A_T1_WAIT:  state <= S_M6A_T1_LATCH;
            S_M6A_T1_LATCH: begin
                pe_A_vec[  0 +: 16] <= dA_reg[ 0]; pe_A_vec[ 16 +: 16] <= dA_reg[ 1];
                pe_A_vec[ 32 +: 16] <= dA_reg[ 2]; pe_A_vec[ 48 +: 16] <= dA_reg[ 3];
                pe_A_vec[ 64 +: 16] <= dA_reg[ 4]; pe_A_vec[ 80 +: 16] <= dA_reg[ 5];
                pe_A_vec[ 96 +: 16] <= dA_reg[ 6]; pe_A_vec[112 +: 16] <= dA_reg[ 7];
                pe_A_vec[128 +: 16] <= dA_reg[ 8]; pe_A_vec[144 +: 16] <= dA_reg[ 9];
                pe_A_vec[160 +: 16] <= dA_reg[10]; pe_A_vec[176 +: 16] <= dA_reg[11];
                pe_A_vec[192 +: 16] <= dA_reg[12]; pe_A_vec[208 +: 16] <= dA_reg[13];
                pe_A_vec[224 +: 16] <= dA_reg[14]; pe_A_vec[240 +: 16] <= dA_reg[15];
                pe_B <= m_rd_data; pe_a_is_vector <= 1; pe_op_mode <= `MODE_MUL;
                state <= S_M6A_T1_WAIT2;
            end
            S_M6A_T1_WAIT2: state <= S_M6A_T1_CAP;
            S_M6A_T1_CAP: begin
                term1_reg[ 0] <= pe_out[  0 +: 16]; term1_reg[ 1] <= pe_out[ 16 +: 16];
                term1_reg[ 2] <= pe_out[ 32 +: 16]; term1_reg[ 3] <= pe_out[ 48 +: 16];
                term1_reg[ 4] <= pe_out[ 64 +: 16]; term1_reg[ 5] <= pe_out[ 80 +: 16];
                term1_reg[ 6] <= pe_out[ 96 +: 16]; term1_reg[ 7] <= pe_out[112 +: 16];
                term1_reg[ 8] <= pe_out[128 +: 16]; term1_reg[ 9] <= pe_out[144 +: 16];
                term1_reg[10] <= pe_out[160 +: 16]; term1_reg[11] <= pe_out[176 +: 16];
                term1_reg[12] <= pe_out[192 +: 16]; term1_reg[13] <= pe_out[208 +: 16];
                term1_reg[14] <= pe_out[224 +: 16]; term1_reg[15] <= pe_out[240 +: 16];
                state <= S_M6A_T2_READ;
            end

            S_M6A_T2_READ: begin
                bank_sel  <= 1;
                m_rd_addr <= B_U_SAFE + t_stride_m + {11'd0, c_grp_m};
                state     <= S_M6A_T2_WAIT;
            end
            S_M6A_T2_WAIT:  state <= S_M6A_T2_LATCH;
            S_M6A_T2_LATCH: begin
                pe_A_vec[  0 +: 16] <= dB_reg[ 0]; pe_A_vec[ 16 +: 16] <= dB_reg[ 1];
                pe_A_vec[ 32 +: 16] <= dB_reg[ 2]; pe_A_vec[ 48 +: 16] <= dB_reg[ 3];
                pe_A_vec[ 64 +: 16] <= dB_reg[ 4]; pe_A_vec[ 80 +: 16] <= dB_reg[ 5];
                pe_A_vec[ 96 +: 16] <= dB_reg[ 6]; pe_A_vec[112 +: 16] <= dB_reg[ 7];
                pe_A_vec[128 +: 16] <= dB_reg[ 8]; pe_A_vec[144 +: 16] <= dB_reg[ 9];
                pe_A_vec[160 +: 16] <= dB_reg[10]; pe_A_vec[176 +: 16] <= dB_reg[11];
                pe_A_vec[192 +: 16] <= dB_reg[12]; pe_A_vec[208 +: 16] <= dB_reg[13];
                pe_A_vec[224 +: 16] <= dB_reg[14]; pe_A_vec[240 +: 16] <= dB_reg[15];
                pe_B <= m_rd_data; pe_a_is_vector <= 1; pe_op_mode <= `MODE_MUL;
                state <= S_M6A_T2_WAIT2;
            end
            S_M6A_T2_WAIT2: state <= S_M6A_T2_CAP;
            S_M6A_T2_CAP: begin
                term2_reg[ 0] <= pe_out[  0 +: 16]; term2_reg[ 1] <= pe_out[ 16 +: 16];
                term2_reg[ 2] <= pe_out[ 32 +: 16]; term2_reg[ 3] <= pe_out[ 48 +: 16];
                term2_reg[ 4] <= pe_out[ 64 +: 16]; term2_reg[ 5] <= pe_out[ 80 +: 16];
                term2_reg[ 6] <= pe_out[ 96 +: 16]; term2_reg[ 7] <= pe_out[112 +: 16];
                term2_reg[ 8] <= pe_out[128 +: 16]; term2_reg[ 9] <= pe_out[144 +: 16];
                term2_reg[10] <= pe_out[160 +: 16]; term2_reg[11] <= pe_out[176 +: 16];
                term2_reg[12] <= pe_out[192 +: 16]; term2_reg[13] <= pe_out[208 +: 16];
                term2_reg[14] <= pe_out[224 +: 16]; term2_reg[15] <= pe_out[240 +: 16];
                state <= S_M6A_HW;
            end

            S_M6A_HW: begin
                bank_sel  <= 1; m_we <= 1;
                m_wr_addr <= A_H_STATE + ({11'd0, s_idx} * {7'd0, ch_m_actual}) + {11'd0, c_grp_m};
                m_wr_data[  0 +: 16] <= sat_add16(term1_reg[ 0], term2_reg[ 0]);
                m_wr_data[ 16 +: 16] <= sat_add16(term1_reg[ 1], term2_reg[ 1]);
                m_wr_data[ 32 +: 16] <= sat_add16(term1_reg[ 2], term2_reg[ 2]);
                m_wr_data[ 48 +: 16] <= sat_add16(term1_reg[ 3], term2_reg[ 3]);
                m_wr_data[ 64 +: 16] <= sat_add16(term1_reg[ 4], term2_reg[ 4]);
                m_wr_data[ 80 +: 16] <= sat_add16(term1_reg[ 5], term2_reg[ 5]);
                m_wr_data[ 96 +: 16] <= sat_add16(term1_reg[ 6], term2_reg[ 6]);
                m_wr_data[112 +: 16] <= sat_add16(term1_reg[ 7], term2_reg[ 7]);
                m_wr_data[128 +: 16] <= sat_add16(term1_reg[ 8], term2_reg[ 8]);
                m_wr_data[144 +: 16] <= sat_add16(term1_reg[ 9], term2_reg[ 9]);
                m_wr_data[160 +: 16] <= sat_add16(term1_reg[10], term2_reg[10]);
                m_wr_data[176 +: 16] <= sat_add16(term1_reg[11], term2_reg[11]);
                m_wr_data[192 +: 16] <= sat_add16(term1_reg[12], term2_reg[12]);
                m_wr_data[208 +: 16] <= sat_add16(term1_reg[13], term2_reg[13]);
                m_wr_data[224 +: 16] <= sat_add16(term1_reg[14], term2_reg[14]);
                m_wr_data[240 +: 16] <= sat_add16(term1_reg[15], term2_reg[15]);
                state <= S_M6A_NEXT;
            end
            S_M6A_NEXT: begin
                pe_a_is_vector <= 0;
                if (s_idx == 4'd15) begin
                    s_idx <= 0;
                    if (c_grp_m == ch_m_last) begin
                        c_grp_m <= 0; bank_sel <= 0; state <= S_M6B_INIT;
                    end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M6A_DA_READ; end
                end else begin s_idx <= s_idx + 4'd1; state <= S_M6A_DA_READ; end
            end

            // --------------------------------------------------------
            // M6b : y = C*h + D*u
            // --------------------------------------------------------
            S_M6B_INIT: begin s_idx <= 0; bank_sel <= 0; state <= S_M6B_RH_READ; end
            S_M6B_RH_READ: begin
                bank_sel  <= 0;
                m_rd_addr <= A_H_STATE + ({11'd0, s_idx} * {7'd0, ch_m_actual}) + {11'd0, c_grp_m};
                state     <= S_M6B_RH_WAIT;
            end
            S_M6B_RH_WAIT:  state <= S_M6B_RH_LATCH;
            S_M6B_RH_LATCH: begin h_reg <= m_rd_data; state <= S_M6B_RC_READ; end

            S_M6B_RC_READ: begin
                bank_sel  <= 1;
                m_rd_addr <= B_X_CONV + t_stride_xp + c_grp_offset;
                state     <= S_M6B_RC_WAIT;
            end
            S_M6B_RC_WAIT:  state <= S_M6B_RC_LATCH;
            S_M6B_RC_LATCH: begin
                pe_A_vec       <= h_reg;
                pe_B           <= {16{m_rd_data[c_lane * 16 +: 16]}};
                C_scalar_reg   <= m_rd_data[c_lane * 16 +: 16];
                pe_a_is_vector <= 1; pe_op_mode <= `MODE_MAC;
                pe_clear       <= (s_idx == 4'd0);
                state          <= S_M6B_RC_WAIT2;
            end
            S_M6B_RC_WAIT2: begin pe_B <= 256'd0; state <= S_M6B_S_NEXT; end
            S_M6B_S_NEXT: begin
                if (s_idx == 4'd15) begin s_idx <= 0; state <= S_M6B_CAP_Y; end
                else begin s_idx <= s_idx + 4'd1; state <= S_M6B_RH_READ; end
            end
            S_M6B_CAP_Y: begin
                y_acc_reg[ 0] <= pe_out[  0 +: 16]; y_acc_reg[ 1] <= pe_out[ 16 +: 16];
                y_acc_reg[ 2] <= pe_out[ 32 +: 16]; y_acc_reg[ 3] <= pe_out[ 48 +: 16];
                y_acc_reg[ 4] <= pe_out[ 64 +: 16]; y_acc_reg[ 5] <= pe_out[ 80 +: 16];
                y_acc_reg[ 6] <= pe_out[ 96 +: 16]; y_acc_reg[ 7] <= pe_out[112 +: 16];
                y_acc_reg[ 8] <= pe_out[128 +: 16]; y_acc_reg[ 9] <= pe_out[144 +: 16];
                y_acc_reg[10] <= pe_out[160 +: 16]; y_acc_reg[11] <= pe_out[176 +: 16];
                y_acc_reg[12] <= pe_out[192 +: 16]; y_acc_reg[13] <= pe_out[208 +: 16];
                y_acc_reg[14] <= pe_out[224 +: 16]; y_acc_reg[15] <= pe_out[240 +: 16];
                state <= S_M6B_DU_READ;
            end
            S_M6B_DU_READ: begin
                bank_sel  <= 1;
                m_rd_addr <= B_U_SAFE + t_stride_m + {11'd0, c_grp_m};
                w_rd_addr <= W_DPARAM_BASE + {11'd0, c_grp_m};
                state     <= S_M6B_DU_WAIT;
            end
            S_M6B_DU_WAIT:  state <= S_M6B_DU_LATCH;
            S_M6B_DU_LATCH: begin
                pe_A_vec <= w_rd_data; pe_B <= m_rd_data;
                pe_a_is_vector <= 1; pe_op_mode <= `MODE_MUL; pe_clear <= 0;
                state <= S_M6B_DU_WAIT2;
            end
            S_M6B_DU_WAIT2: state <= S_M6B_DU_CAP;
            S_M6B_DU_CAP: begin
                du_reg[ 0] <= pe_out[  0 +: 16]; du_reg[ 1] <= pe_out[ 16 +: 16];
                du_reg[ 2] <= pe_out[ 32 +: 16]; du_reg[ 3] <= pe_out[ 48 +: 16];
                du_reg[ 4] <= pe_out[ 64 +: 16]; du_reg[ 5] <= pe_out[ 80 +: 16];
                du_reg[ 6] <= pe_out[ 96 +: 16]; du_reg[ 7] <= pe_out[112 +: 16];
                du_reg[ 8] <= pe_out[128 +: 16]; du_reg[ 9] <= pe_out[144 +: 16];
                du_reg[10] <= pe_out[160 +: 16]; du_reg[11] <= pe_out[176 +: 16];
                du_reg[12] <= pe_out[192 +: 16]; du_reg[13] <= pe_out[208 +: 16];
                du_reg[14] <= pe_out[224 +: 16]; du_reg[15] <= pe_out[240 +: 16];
                state <= S_M6B_WRITE;
            end
            S_M6B_WRITE: begin
                bank_sel  <= 0; m_we <= 1;
                m_wr_addr <= B_Y_SSM + t_stride_m + {11'd0, c_grp_m};
                m_wr_data[  0 +: 16] <= sat_add16(y_acc_reg[ 0], du_reg[ 0]);
                m_wr_data[ 16 +: 16] <= sat_add16(y_acc_reg[ 1], du_reg[ 1]);
                m_wr_data[ 32 +: 16] <= sat_add16(y_acc_reg[ 2], du_reg[ 2]);
                m_wr_data[ 48 +: 16] <= sat_add16(y_acc_reg[ 3], du_reg[ 3]);
                m_wr_data[ 64 +: 16] <= sat_add16(y_acc_reg[ 4], du_reg[ 4]);
                m_wr_data[ 80 +: 16] <= sat_add16(y_acc_reg[ 5], du_reg[ 5]);
                m_wr_data[ 96 +: 16] <= sat_add16(y_acc_reg[ 6], du_reg[ 6]);
                m_wr_data[112 +: 16] <= sat_add16(y_acc_reg[ 7], du_reg[ 7]);
                m_wr_data[128 +: 16] <= sat_add16(y_acc_reg[ 8], du_reg[ 8]);
                m_wr_data[144 +: 16] <= sat_add16(y_acc_reg[ 9], du_reg[ 9]);
                m_wr_data[160 +: 16] <= sat_add16(y_acc_reg[10], du_reg[10]);
                m_wr_data[176 +: 16] <= sat_add16(y_acc_reg[11], du_reg[11]);
                m_wr_data[192 +: 16] <= sat_add16(y_acc_reg[12], du_reg[12]);
                m_wr_data[208 +: 16] <= sat_add16(y_acc_reg[13], du_reg[13]);
                m_wr_data[224 +: 16] <= sat_add16(y_acc_reg[14], du_reg[14]);
                m_wr_data[240 +: 16] <= sat_add16(y_acc_reg[15], du_reg[15]);
                state <= S_M6B_NEXT;
            end
            S_M6B_NEXT: begin
                pe_a_is_vector <= 0;
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp_m <= 0; bank_sel <= 1; state <= S_M7_RY_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; s_idx <= 0; state <= S_M6A_DA_READ; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M6B_INIT; end
            end

            // --------------------------------------------------------
            // M7 : y_gated = y_ssm * SiLU(z)
            // --------------------------------------------------------
            S_M7_RY_READ: begin
                bank_sel  <= 1;
                m_rd_addr <= B_Y_SSM + t_stride_m + {11'd0, c_grp_m};
                state     <= S_M7_RY_WAIT;
            end
            S_M7_RY_WAIT:  state <= S_M7_RY_LATCH;
            S_M7_RY_LATCH: begin y_reg <= m_rd_data; state <= S_M7_RZ_READ; end

            S_M7_RZ_READ: begin
                bank_sel  <= 0;
                m_rd_addr <= A_Z_GATE + t_stride_m + {11'd0, c_grp_m};
                state     <= S_M7_RZ_WAIT;
            end
            S_M7_RZ_WAIT:  state <= S_M7_RZ_LATCH;
            S_M7_RZ_LATCH: begin
                pe_A_vec <= y_reg;
                pe_B[  0 +: 16] <= silu_o[ 0]; pe_B[ 16 +: 16] <= silu_o[ 1];
                pe_B[ 32 +: 16] <= silu_o[ 2]; pe_B[ 48 +: 16] <= silu_o[ 3];
                pe_B[ 64 +: 16] <= silu_o[ 4]; pe_B[ 80 +: 16] <= silu_o[ 5];
                pe_B[ 96 +: 16] <= silu_o[ 6]; pe_B[112 +: 16] <= silu_o[ 7];
                pe_B[128 +: 16] <= silu_o[ 8]; pe_B[144 +: 16] <= silu_o[ 9];
                pe_B[160 +: 16] <= silu_o[10]; pe_B[176 +: 16] <= silu_o[11];
                pe_B[192 +: 16] <= silu_o[12]; pe_B[208 +: 16] <= silu_o[13];
                pe_B[224 +: 16] <= silu_o[14]; pe_B[240 +: 16] <= silu_o[15];
                pe_a_is_vector <= 1; pe_op_mode <= `MODE_MUL;
                state          <= S_M7_PE_WAIT2;
            end
            S_M7_PE_WAIT2: state <= S_M7_WRITE;
            S_M7_WRITE: begin
                bank_sel  <= 0; m_we <= 1;
                m_wr_addr <= B_Y_SSM + t_stride_m + {11'd0, c_grp_m};
                m_wr_data <= pe_out; state <= S_M7_NEXT;
            end
            S_M7_NEXT: begin
                pe_a_is_vector <= 0;
                if (c_grp_m == ch_m_last) begin
                    c_grp_m <= 0;
                    if (t_cnt == t_last) begin
                        t_cnt <= 0; c_grp <= 0; mac_idx <= 0; substep <= 0; bank_sel <= 1; state <= S_M8_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; state <= S_M7_RY_READ; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M7_RY_READ; end
            end

            // --------------------------------------------------------
            // M8 : out_proj  d_inner -> d_out
            // --------------------------------------------------------
            S_M8_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= B_Y_SSM + t_stride_m + {7'd0, mac_idx[7:4]};
                        w_rd_addr <= W_OUTPROJ_BASE + ({11'd0, c_grp} * {7'd0, d_inner}) + {7'd0, mac_idx};
                        pe_A <= 16'sd0; pe_a_is_vector <= 0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        pe_A       <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B       <= w_rd_data;
                        pe_op_mode <= `MODE_MAC;
                        pe_clear   <= (mac_idx == 8'd0);
                        if (mac_idx == d_inner_last) state <= S_M8_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end
            S_M8_WAIT:  state <= S_M8_WRITE;
            S_M8_WRITE: begin
                bank_sel  <= 1; m_we <= 1;
                m_wr_addr <= A_MAMBA_OUT + t_stride_out + {12'd0, c_grp};
                m_wr_data <= pe_out; state <= S_M8_NEXT;
            end
            S_M8_NEXT: begin
                mac_idx <= 0; substep <= 0;
                if (c_grp == ch_out_last[2:0]) begin  // ch_out_last groups
                    c_grp <= 0;
                    if (t_cnt == t_last) begin
                        done_mamba <= 1;
                        t_cnt <= 0; c_grp <= 0; substep <= 0;
                        bank_sel  <= 0;
                        c_rd_addr <= C_INC_SCALE;
                        state     <= S_FIN_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; bank_sel <= 1; state <= S_M8_MAC; end
                end else begin c_grp <= c_grp + 3'd1; bank_sel <= 1; state <= S_M8_MAC; end
            end

            // --------------------------------------------------------
            // PHASE 4 : Final  bn_relu(inception + mamba_out)
            // c_grp 0..CH_OUT-1. For blk4: c_grp encodes branch[2:1] + sub[0].
            // --------------------------------------------------------
            S_FIN_READ: begin
                bank_sel  <= fin_branch_bank;
                m_rd_addr <= fin_branch_base
                           + ({5'd0, t_cnt} * br_dim_groups)
                           + {14'd0, fin_sub};
                c_rd_addr <= C_INC_SCALE + {12'd0, c_grp};
                state     <= S_FIN_WAIT;
            end
            S_FIN_WAIT: state <= S_FIN_MUL;
            S_FIN_MUL: begin
                incep_reg <= m_rd_data;
                m_wr_data <= c_rd_data;           // holds scale word
                c_rd_addr <= C_INC_SHIFT + {12'd0, c_grp};
                state     <= S_FIN_WAIT2;
            end
            S_FIN_WAIT2: begin
                bank_sel  <= 0;
                m_rd_addr <= A_MAMBA_OUT + t_stride_out + {12'd0, c_grp};
                state     <= S_FIN_READ_M;
            end
            S_FIN_READ_M: state <= S_FIN_WAIT_M;
            S_FIN_WAIT_M: state <= S_FIN_WRITE;
            S_FIN_WRITE: begin
                bank_sel  <= (c_grp == 3'd0) ? 1'b0 : 1'b1;
                m_we      <= 1;
                m_wr_addr <= (c_grp == 3'd0) ? (B_FINAL_OUT + t_stride_out)
                                              : (A_FINAL_OUT + t_stride_out + {12'd0, c_grp});
                m_wr_data[  0 +: 16] <= bn_relu(sat_add16(incep_reg[  0 +: 16], m_rd_data[  0 +: 16]), m_wr_data[  0 +: 16], c_rd_data[  0 +: 16]);
                m_wr_data[ 16 +: 16] <= bn_relu(sat_add16(incep_reg[ 16 +: 16], m_rd_data[ 16 +: 16]), m_wr_data[ 16 +: 16], c_rd_data[ 16 +: 16]);
                m_wr_data[ 32 +: 16] <= bn_relu(sat_add16(incep_reg[ 32 +: 16], m_rd_data[ 32 +: 16]), m_wr_data[ 32 +: 16], c_rd_data[ 32 +: 16]);
                m_wr_data[ 48 +: 16] <= bn_relu(sat_add16(incep_reg[ 48 +: 16], m_rd_data[ 48 +: 16]), m_wr_data[ 48 +: 16], c_rd_data[ 48 +: 16]);
                m_wr_data[ 64 +: 16] <= bn_relu(sat_add16(incep_reg[ 64 +: 16], m_rd_data[ 64 +: 16]), m_wr_data[ 64 +: 16], c_rd_data[ 64 +: 16]);
                m_wr_data[ 80 +: 16] <= bn_relu(sat_add16(incep_reg[ 80 +: 16], m_rd_data[ 80 +: 16]), m_wr_data[ 80 +: 16], c_rd_data[ 80 +: 16]);
                m_wr_data[ 96 +: 16] <= bn_relu(sat_add16(incep_reg[ 96 +: 16], m_rd_data[ 96 +: 16]), m_wr_data[ 96 +: 16], c_rd_data[ 96 +: 16]);
                m_wr_data[112 +: 16] <= bn_relu(sat_add16(incep_reg[112 +: 16], m_rd_data[112 +: 16]), m_wr_data[112 +: 16], c_rd_data[112 +: 16]);
                m_wr_data[128 +: 16] <= bn_relu(sat_add16(incep_reg[128 +: 16], m_rd_data[128 +: 16]), m_wr_data[128 +: 16], c_rd_data[128 +: 16]);
                m_wr_data[144 +: 16] <= bn_relu(sat_add16(incep_reg[144 +: 16], m_rd_data[144 +: 16]), m_wr_data[144 +: 16], c_rd_data[144 +: 16]);
                m_wr_data[160 +: 16] <= bn_relu(sat_add16(incep_reg[160 +: 16], m_rd_data[160 +: 16]), m_wr_data[160 +: 16], c_rd_data[160 +: 16]);
                m_wr_data[176 +: 16] <= bn_relu(sat_add16(incep_reg[176 +: 16], m_rd_data[176 +: 16]), m_wr_data[176 +: 16], c_rd_data[176 +: 16]);
                m_wr_data[192 +: 16] <= bn_relu(sat_add16(incep_reg[192 +: 16], m_rd_data[192 +: 16]), m_wr_data[192 +: 16], c_rd_data[192 +: 16]);
                m_wr_data[208 +: 16] <= bn_relu(sat_add16(incep_reg[208 +: 16], m_rd_data[208 +: 16]), m_wr_data[208 +: 16], c_rd_data[208 +: 16]);
                m_wr_data[224 +: 16] <= bn_relu(sat_add16(incep_reg[224 +: 16], m_rd_data[224 +: 16]), m_wr_data[224 +: 16], c_rd_data[224 +: 16]);
                m_wr_data[240 +: 16] <= bn_relu(sat_add16(incep_reg[240 +: 16], m_rd_data[240 +: 16]), m_wr_data[240 +: 16], c_rd_data[240 +: 16]);
                state <= S_FIN_NEXT;
            end
            S_FIN_NEXT: begin
                if (t_cnt == t_last) begin
                    t_cnt <= 0;
                    if (c_grp == ch_out_last[2:0]) begin
                        done_all <= 1; state <= S_DONE;
                    end else begin
                        c_grp    <= c_grp + 3'd1;
                        bank_sel <= 1;
                        state    <= S_FIN_READ;
                    end
                end else begin t_cnt <= t_cnt + 10'd1; state <= S_FIN_READ; end
            end

            S_DONE: begin
                done_phase1 <= 1; done_inception <= 1; done_mamba <= 1; done_all <= 1;
                state       <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // ================================================================
    // Helper functions (Verilog-2001)
    // ================================================================
    function signed [15:0] sat_add16;
        input signed [15:0] a;
        input signed [15:0] b;
        reg signed [16:0] s;
        begin
            s = {a[15], a} + {b[15], b};
            if      (s >  17'sd32767) sat_add16 = 16'sh7FFF;
            else if (s < -17'sd32768) sat_add16 = 16'sh8000;
            else                      sat_add16 = s[15:0];
        end
    endfunction

    function signed [15:0] bn_relu;
        input signed [15:0] raw;
        input signed [15:0] scale;
        input signed [15:0] shift;
        reg signed [31:0] mul_raw;
        reg signed [31:0] mul_shifted;
        reg signed [16:0] s;
        reg signed [15:0] bn_out;
        begin
            mul_raw     = raw * scale;
            mul_shifted = mul_raw >>> `FRAC_BITS;
            if      (mul_shifted >  32'sd32767) bn_out = 16'sh7FFF;
            else if (mul_shifted < -32'sd32768) bn_out = 16'sh8000;
            else                                bn_out = mul_shifted[15:0];
            s = {bn_out[15], bn_out} + {shift[15], shift};
            if      (s >  17'sd32767) bn_out = 16'sh7FFF;
            else if (s < -17'sd32768) bn_out = 16'sh8000;
            else                      bn_out = s[15:0];
            bn_relu = bn_out[15] ? 16'sd0 : bn_out;
        end
    endfunction

endmodule