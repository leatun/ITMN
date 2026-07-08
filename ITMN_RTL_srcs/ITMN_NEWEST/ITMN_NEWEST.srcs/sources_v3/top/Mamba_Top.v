`include "_parameter.v"

// ============================================================================
// Mamba_Top — per-timestep chained pipeline.
//   Sequence per t: RMSNorm → M1A → M1B → M2 → M3 → M4 → M5 → M6 → M7 → M8
//   H_RegFile is zero-initialized once before the T loop starts.
//   All intermediates use compact PT_* slot addresses (see _parameter.v).
//   INPUT and MAMBA_OUT alias the same bulk region (INPUT[t] dies before
//   MAMBA_OUT[t] is written).
// DMA targets: 0=ram_main, 2=ram_weight, 3=ram_const.
// ============================================================================

module Mamba_Top (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    output reg          done_stage,
    output reg          done_all,

    input  wire [3:0]   run_stage,   // ignored — always full pipeline
    input  wire [9:0]   T_MAX,
    input  wire [3:0]   CH_OUT,
    input  wire [3:0]   CH_M,
    input  wire [3:0]   DT_RANK,

    input  wire         dma_write_en,
    input  wire [1:0]   dma_target,
    input  wire [14:0]  dma_addr,
    input  wire [255:0] dma_wdata,

    input  wire         dma_read_en,
    input  wire [1:0]   dma_rtarget,
    input  wire [14:0]  dma_raddr,
    output wire [255:0] dma_rdata
);

    // ----- Derived dimensions -----
    wire [4:0]  ch_m_act = (CH_M == 4'd0) ? 5'd16 : {1'b0, CH_M};
    wire [8:0]  d_inner  = {ch_m_act, 4'b0};
    wire [7:0]  d_model  = {CH_OUT, 4'b0};
    wire [9:0]  t_last   = T_MAX - 10'd1;
    wire [2:0]  log2_dm  = (CH_OUT == 4'd4) ? 3'd2 : 3'd3;
    wire [4:0]  rn_total_shift = {2'b0, log2_dm} + 5'd19;
    localparam [4:0] XP_OUT_GRP = 5'd3;

    // ----- PT slot bases -----
    localparam [14:0] PT_X_NORM        = `PT_X_NORM;
    localparam [14:0] PT_X_PROJ        = `PT_X_PROJ;
    localparam [14:0] PT_Y_GATED       = `PT_Y_GATED;
    localparam [14:0] PT_X_CONV        = `PT_X_CONV;
    localparam [14:0] PT_U             = `PT_U;
    localparam [14:0] PT_Y_SSM         = `PT_Y_SSM;
    localparam [14:0] PT_DELTA         = `PT_DELTA;
    localparam [14:0] PT_Z_GATE        = `PT_Z_GATE;
    localparam [14:0] PT_X_INNER_CIRC  = `PT_X_INNER_CIRC;
    localparam [14:0] PT_INPUT         = `PT_INPUT;
    localparam [14:0] PT_MAMBA_OUT     = `PT_MAMBA_OUT;
    localparam [14:0] W_INPROJ_X_BASE  = `W_INPROJ_X_BASE;
    localparam [14:0] W_INPROJ_Z_BASE  = `W_INPROJ_Z_BASE;
    localparam [14:0] W_OUTPROJ_BASE   = `W_OUTPROJ_BASE;
    localparam [14:0] W_DW_BASE        = `W_DW_BASE;
    localparam [14:0] W_XPROJ_BASE     = `W_XPROJ_BASE;
    localparam [14:0] W_DTPROJ_BASE    = `W_DTPROJ_BASE;
    localparam [14:0] W_A_BASE         = `W_A_BASE;
    localparam [14:0] C_W_NORM_BASE    = `C_W_NORM_BASE;
    localparam [14:0] C_B_DW_BASE      = `C_B_DW_BASE;
    localparam [14:0] C_B_DT_BASE      = `C_B_DT_BASE;
    localparam [14:0] C_D_PARAM_BASE   = `C_D_PARAM_BASE;

    // ----- FSM state encoding -----
    localparam [6:0] S_IDLE          = 7'd0;
    localparam [6:0] S_H_INIT        = 7'd1;
    // MAC stages (M1A/M1B/M4/M8)
    localparam [6:0] S_PREFETCH      = 7'd2;
    localparam [6:0] S_WAIT          = 7'd3;
    localparam [6:0] S_MAC           = 7'd4;
    localparam [6:0] S_NEXT          = 7'd5;
    // RMSNorm
    localparam [6:0] S_RN_SQ_PREF    = 7'd6;
    localparam [6:0] S_RN_SQ_WAIT    = 7'd7;
    localparam [6:0] S_RN_SQ_MAC     = 7'd8;
    localparam [6:0] S_RN_SQ_FINAL   = 7'd9;
    localparam [6:0] S_RN_SQ_DONE    = 7'd10;
    localparam [6:0] S_RN_RSQ_WAIT   = 7'd11;
    localparam [6:0] S_RN_S_LATCH    = 7'd12;
    localparam [6:0] S_RN_AP_PREF    = 7'd13;
    localparam [6:0] S_RN_AP_WAIT    = 7'd14;
    localparam [6:0] S_RN_AP_MUL1    = 7'd15;
    localparam [6:0] S_RN_AP_WAIT1   = 7'd16;
    localparam [6:0] S_RN_AP_MUL2    = 7'd17;
    localparam [6:0] S_RN_AP_WAIT2   = 7'd18;
    localparam [6:0] S_RN_AP_WRITE   = 7'd19;
    localparam [6:0] S_RN_AP_NEXT    = 7'd20;
    // M2
    localparam [6:0] S_M2_PREF       = 7'd21;
    localparam [6:0] S_M2_WAIT       = 7'd22;
    localparam [6:0] S_M2_TAP        = 7'd23;
    localparam [6:0] S_M2_FINAL      = 7'd24;
    localparam [6:0] S_M2_BIAS_PREF  = 7'd25;
    localparam [6:0] S_M2_BIAS_WAIT  = 7'd26;
    localparam [6:0] S_M2_BIAS_ADD   = 7'd27;
    localparam [6:0] S_M2_BIAS_LATCH = 7'd28;
    localparam [6:0] S_M2_WRITE      = 7'd29;
    localparam [6:0] S_M2_NEXT       = 7'd30;
    // M3
    localparam [6:0] S_M3_PREF       = 7'd31;
    localparam [6:0] S_M3_WAIT       = 7'd32;
    localparam [6:0] S_M3_WRITE      = 7'd33;
    localparam [6:0] S_M3_NEXT       = 7'd34;
    // M5
    localparam [6:0] S_M5_DT_READ    = 7'd35;
    localparam [6:0] S_M5_DT_WAIT    = 7'd36;
    localparam [6:0] S_M5_DT_LATCH   = 7'd37;
    localparam [6:0] S_M5_W_PREF     = 7'd38;
    localparam [6:0] S_M5_W_WAIT     = 7'd39;
    localparam [6:0] S_M5_MAC        = 7'd40;
    localparam [6:0] S_M5_FINAL      = 7'd41;
    localparam [6:0] S_M5_BIAS_PREF  = 7'd42;
    localparam [6:0] S_M5_BIAS_WAIT  = 7'd43;
    localparam [6:0] S_M5_BIAS_ADD   = 7'd44;
    localparam [6:0] S_M5_BIAS_LATCH = 7'd45;
    localparam [6:0] S_M5_WRITE      = 7'd46;
    localparam [6:0] S_M5_NEXT       = 7'd47;
    // M7
    localparam [6:0] S_M7_Z_PREF     = 7'd48;
    localparam [6:0] S_M7_Z_WAIT     = 7'd49;
    localparam [6:0] S_M7_Z_LATCH    = 7'd50;
    localparam [6:0] S_M7_Y_PREF     = 7'd51;
    localparam [6:0] S_M7_Y_WAIT     = 7'd52;
    localparam [6:0] S_M7_MUL        = 7'd53;
    localparam [6:0] S_M7_LATCH      = 7'd54;
    localparam [6:0] S_M7_WRITE      = 7'd55;
    localparam [6:0] S_M7_NEXT       = 7'd56;
    // M6 SSM scan
    localparam [6:0] S_M6_LOAD_W0_PREF = 7'd95;
    localparam [6:0] S_M6_LOAD_W0_WAIT = 7'd96;
    localparam [6:0] S_M6_LOAD_W0_LATCH= 7'd97;
    // Registered-output latch states — feed downstream registers from
    // cl_out_vec (registered PE output) instead of cl_out_next_vec (comb),
    // breaking URAM→mult→sat→wr comb chains that fail at 100 MHz.
    localparam [6:0] S_MAC_LATCH        = 7'd98;
    localparam [6:0] S_M6_T1_WAIT       = 7'd99;
    localparam [6:0] S_M6_T2_WAIT       = 7'd100;
    localparam [6:0] S_M6_DU_WAIT       = 7'd101;
    // MAC prefill: capture pair 0 into cl_in_*_vec regs so PE.in_H no longer
    // needs the URAM→y_lane_sel→mac_bypass comb chain. PE stays IDLE for
    // this cycle; MAC starts on next edge with clear_acc=1.
    localparam [6:0] S_MAC_LOAD         = 7'd102;
    localparam [6:0] S_M6_LOAD_B_PREF  = 7'd57;
    localparam [6:0] S_M6_LOAD_B_WAIT  = 7'd58;
    localparam [6:0] S_M6_LOAD_B_LATCH = 7'd59;
    localparam [6:0] S_M6_LOAD_C_PREF  = 7'd60;
    localparam [6:0] S_M6_LOAD_C_WAIT  = 7'd61;
    localparam [6:0] S_M6_LOAD_C_LATCH = 7'd62;
    localparam [6:0] S_M6_LOAD_DT_PREF  = 7'd63;
    localparam [6:0] S_M6_LOAD_DT_WAIT  = 7'd64;
    localparam [6:0] S_M6_LOAD_DT_LATCH = 7'd65;
    localparam [6:0] S_M6_LOAD_U_PREF   = 7'd66;
    localparam [6:0] S_M6_LOAD_U_WAIT   = 7'd67;
    localparam [6:0] S_M6_LOAD_U_LATCH  = 7'd68;
    localparam [6:0] S_M6_LOAD_D_PREF   = 7'd69;
    localparam [6:0] S_M6_LOAD_D_WAIT   = 7'd70;
    localparam [6:0] S_M6_LOAD_D_LATCH  = 7'd71;
    localparam [6:0] S_M6_A_PREF        = 7'd72;
    localparam [6:0] S_M6_A_WAIT        = 7'd73;
    // MUL2 fusion: DA_MUL + DB_MUL merged into DAB_MUL2 (single MAMBA_PE_MUL2 op).
    // EXP_WAIT lets PE register cl_out_vec / cl_out_vec2 so DAB_LATCH can feed
    // Exp_LUT from registered output (breaks the ~13 ns URAM→mult→LUT chain).
    localparam [6:0] S_M6_DAB_MUL2      = 7'd74;
    localparam [6:0] S_M6_EXP_WAIT      = 7'd75;
    localparam [6:0] S_M6_DAB_LATCH     = 7'd76;
    // (7'd77 = old S_M6_DB_LATCH slot, now unused)
    localparam [6:0] S_M6_H_PREF        = 7'd78;
    localparam [6:0] S_M6_H_WAIT        = 7'd79;
    localparam [6:0] S_M6_T1_MUL        = 7'd80;
    localparam [6:0] S_M6_T1_LATCH      = 7'd81;
    localparam [6:0] S_M6_T2_MUL        = 7'd82;
    localparam [6:0] S_M6_T2_LATCH      = 7'd83;
    localparam [6:0] S_M6_H_ADD         = 7'd84;
    localparam [6:0] S_M6_H_WRITE       = 7'd85;
    localparam [6:0] S_M6_Y_MAC         = 7'd86;
    localparam [6:0] S_M6_Y_WAIT        = 7'd87;
    localparam [6:0] S_M6_Y_LATCH       = 7'd88;
    localparam [6:0] S_M6_DU_MUL        = 7'd89;
    localparam [6:0] S_M6_DU_LATCH      = 7'd90;
    localparam [6:0] S_M6_FINALIZE      = 7'd91;
    localparam [6:0] S_M6_LANE_NEXT     = 7'd92;
    localparam [6:0] S_M6_WRITE_GRP     = 7'd93;
    localparam [6:0] S_M6_GRP_NEXT      = 7'd94;
    localparam [6:0] S_DONE             = 7'd127;

    // ----- Registers -----
    reg [6:0]   state;
    reg [3:0]   cur_stage;
    reg [9:0]   t_cnt;
    reg [4:0]   c_out_grp;
    reg [8:0]   c_in_cnt;
    reg [4:0]   rn_grp;
    reg [1:0]   tap_cnt;
    reg [4:0]   m5_w_cnt;
    reg [255:0] silu_z_reg;
    reg [255:0] m5_dt_word_reg;
    reg [3:0]   m5_dt_lane;
    reg [15:0]  S_reg;
    reg [8:0]   h_init_cnt;

    // M6 state
    reg [3:0]   m6_lane;
    reg [255:0] m6_delta_word, m6_u_word, m6_w0_reg, m6_w1_reg, m6_w2_reg, m6_D_word;
    reg [255:0] m6_dA_reg, m6_dB_reg, m6_t1_reg, m6_t2_reg, m6_ssm_grp_acc;
    reg signed [15:0] m6_y_ch_reg, m6_du_reg;

    // X_PROJ layout is raw-concat [dt(dt_rank), B(d_state), C(d_state), pad]
    // packed 16-per-word.  Since dt_rank (=4) is not a multiple of 16,
    // B and C span word boundaries.  Shuffle 3 raw words → aligned B/C.
    wire [8:0]  m6_dt_shift = {DT_RANK, 4'b0};        // DT_RANK * 16 bits
    wire [8:0]  m6_dt_comp  = 9'd256 - m6_dt_shift;
    wire [255:0] m6_B_word  = (m6_w1_reg << m6_dt_comp) | (m6_w0_reg >> m6_dt_shift);
    wire [255:0] m6_C_word  = (m6_w2_reg << m6_dt_comp) | (m6_w1_reg >> m6_dt_shift);

    // ----- Memory wires -----
    reg  [14:0]  m_rd_addr;
    reg          m_we;
    reg  [14:0]  m_wr_addr;
    reg  [255:0] m_wr_data;
    reg  [14:0]  w_rd_addr, w_rd_addr2;
    wire [255:0] m_rd_data, w_rd_data, w_rd_data2, mem_dma_rdata;

    // ----- Cluster wires -----
    reg  [2:0]                cl_op_mode;
    reg                       cl_clear_acc;
    reg  [16*`DATA_W-1:0]     cl_in_W1_vec, cl_in_H_ext, cl_in_W2_vec, cl_in_X_vec;
    wire [16*`DATA_W-1:0]     cl_out_vec, cl_out_next_vec;
    wire [16*`DATA_W-1:0]     cl_out_vec2, cl_out_next_vec2;
    wire [16*`ACC_W-1:0]      cl_acc_raw_vec;
    wire [`DATA_W+4:0]        cl_y_reduce_out;

    // Register pipeline for PE inputs — cl_in_*_vec captured every S_MAC/LOAD
    // cycle. Breaks URAM → y_lane_sel mux → mac_bypass mux → DSP comb chain
    // that fails at 100 MHz. mac_bypass replaced by S_MAC_LOAD prefill state.
    wire [255:0] cl_in_W1_eff = cl_in_W1_vec;
    wire [255:0] cl_in_H_eff  = cl_in_H_ext;
    wire [255:0] cl_in_W2_eff = cl_in_W2_vec;
    wire [255:0] cl_in_X_eff  = cl_in_X_vec;

    wire [3:0]   y_lane_sel  = c_in_cnt[3:0];
    wire [15:0]  y_scalar    = m_rd_data[y_lane_sel*16 +: 16];
    wire [255:0] y_broadcast = {16{y_scalar}};
    wire [3:0]   y_lane_sel2 = c_in_cnt[3:0] + 4'd1;
    wire [15:0]  y_scalar2   = m_rd_data[y_lane_sel2*16 +: 16];
    wire [255:0] y_broadcast2 = {16{y_scalar2}};

    // ----- Const_Storage wires -----
    reg  [14:0]  const_rd_addr_r;
    wire [255:0] const_rd_data;
    reg  [12:0]  rsqrt_idx_r;
    wire [15:0]  rsqrt_data;
    wire [255:0] silu_out_w, sp_out_w, exp_out_w, const_dma_rdata;

    wire [255:0] silu_in_drv, sp_in_drv, exp_in_drv;

    assign dma_rdata = (dma_rtarget == 2'd3) ? const_dma_rdata : mem_dma_rdata;

    // ----- Submodules -----
    Memory_System u_mem (
        .clk              (clk),
        .reset            (rst),
        .core_read_addr   (m_rd_addr),
        .core_read_data   (m_rd_data),
        .core_write_en    (m_we),
        .core_write_addr  (m_wr_addr),
        .core_write_data  (m_wr_data),
        .weight_read_addr (w_rd_addr),
        .weight_read_data (w_rd_data),
        .weight_read_addr2(w_rd_addr2),
        .weight_read_data2(w_rd_data2),
        .dma_write_en     (dma_write_en),
        .dma_target       (dma_target),
        .dma_addr         (dma_addr),
        .dma_wdata        (dma_wdata),
        .dma_read_en      (dma_read_en),
        .dma_rtarget      (dma_rtarget),
        .dma_raddr        (dma_raddr),
        .dma_rdata        (mem_dma_rdata)
    );

    Const_Storage u_const (
        .clk             (clk),
        .silu_in_flat    (silu_in_drv),
        .sp_in_flat      (sp_in_drv),
        .exp_in_flat     (exp_in_drv),
        .silu_out_flat   (silu_out_w),
        .sp_out_flat     (sp_out_w),
        .exp_out_flat    (exp_out_w),
        .rsqrt_idx       (rsqrt_idx_r),
        .rsqrt_data      (rsqrt_data),
        .dma_write_en    (dma_write_en),
        .dma_target      (dma_target),
        .dma_addr        (dma_addr),
        .dma_wdata       (dma_wdata),
        .const_read_addr (const_rd_addr_r),
        .const_read_data (const_rd_data),
        .dma_read_en     (dma_read_en),
        .dma_rtarget     (dma_rtarget),
        .dma_raddr       (dma_raddr),
        .dma_rdata_const (const_dma_rdata)
    );

    // M6 drives H_RegFile; other stages tie to 0/idle.
    reg                       m6_h_from_rf_r;
    reg  [8:0]                m6_h_rd_addr_r;
    reg                       m6_h_wr_en_r;
    reg  [8:0]                m6_h_wr_addr_r;
    reg                       m6_h_wr_from_pe_r;
    reg  [16*`DATA_W-1:0]     m6_h_wr_data_ext_r;

    // Active in M6 and H_INIT stages
    wire h_ctrl_active = (cur_stage == 4'd6) || (state == S_H_INIT);
    wire                      cl_h_from_rf_w     = h_ctrl_active ? m6_h_from_rf_r     : 1'b0;
    wire [8:0]                cl_h_rd_addr_w     = h_ctrl_active ? m6_h_rd_addr_r     : 9'b0;
    wire                      cl_h_wr_en_w       = h_ctrl_active ? m6_h_wr_en_r       : 1'b0;
    wire [8:0]                cl_h_wr_addr_w     = h_ctrl_active ? m6_h_wr_addr_r     : 9'b0;
    wire                      cl_h_wr_from_pe_w  = h_ctrl_active ? m6_h_wr_from_pe_r  : 1'b0;
    wire [16*`DATA_W-1:0]     cl_h_wr_data_ext_w = h_ctrl_active ? m6_h_wr_data_ext_r : 256'b0;

    M_Cluster #(.H_ADDR_W(9), .H_DEPTH(256)) u_mc (
        .clk          (clk),
        .rst          (rst),
        .op_mode      (cl_op_mode),
        .clear_acc    (cl_clear_acc),
        .in_W1_vec    (cl_in_W1_eff),
        .in_H_ext     (cl_in_H_eff),
        .in_W2_vec    (cl_in_W2_eff),
        .in_X_vec     (cl_in_X_eff),
        .h_from_rf    (cl_h_from_rf_w),
        .h_rd_addr    (cl_h_rd_addr_w),
        .h_wr_en      (cl_h_wr_en_w),
        .h_wr_addr    (cl_h_wr_addr_w),
        .h_wr_from_pe (cl_h_wr_from_pe_w),
        .h_wr_data_ext(cl_h_wr_data_ext_w),
        .out_vec      (cl_out_vec),
        .acc_raw_vec  (cl_acc_raw_vec),
        .y_reduce_out (cl_y_reduce_out),
        .out_next_vec (cl_out_next_vec),
        .out_vec2     (cl_out_vec2),
        .out_next_vec2(cl_out_next_vec2)
    );

    wire signed [`ACC_W+4:0] sum_d_wide;
    Reduce16Wide u_rw (
        .in_vec (cl_acc_raw_vec),
        .out_sum(sum_d_wide)
    );

    // RMSNorm sum → mean → rsqrt → S
    wire signed [`ACC_W+4:0] mean_i_signed = sum_d_wide >>> rn_total_shift;
    wire [12:0] mean_i_clip =
        (mean_i_signed < 0)         ? 13'd0    :
        (mean_i_signed > 45'sd8191) ? 13'd8191 :
                                       mean_i_signed[12:0];
    wire [255:0] S_broadcast = {16{S_reg}};

    // Combinational LUT input drivers (latched at same cycle as capture)
    assign silu_in_drv = (state == S_M3_WRITE  ) ? m_rd_data :
                         (state == S_M7_Z_LATCH) ? m_rd_data :
                                                    256'b0;
    assign sp_in_drv   = (state == S_M5_WRITE  ) ? cl_out_vec : 256'b0;
    // Registered feed (cl_out_vec) breaks URAM→mult→LUT→dA_reg comb chain.
    // Active in DAB_LATCH: PE has registered dt*A into cl_out_vec at previous
    // edge; LUT only sees short CLK-Q + ROM comb path.
    assign exp_in_drv  = (state == S_M6_DAB_LATCH) ? cl_out_vec : 256'b0;

    // ============================================================
    //  MAC address helpers (M1A / M1B / M4 / M8)
    // ============================================================
    wire [8:0]  mac_len =
        (cur_stage == 4'd4) ? d_inner :
        (cur_stage == 4'd8) ? d_inner :
                              {1'b0, d_model};
    wire [4:0]  mac_grp_count =
        (cur_stage == 4'd8) ? {1'b0, CH_OUT} :
        (cur_stage == 4'd4) ? XP_OUT_GRP :
                              ch_m_act;
    wire [14:0] mac_in_base =
        (cur_stage == 4'd4) ? PT_U :
        (cur_stage == 4'd8) ? PT_Y_GATED :
                              PT_X_NORM;
    wire [14:0] mac_w_base =
        (cur_stage == 4'd0) ? W_INPROJ_X_BASE :
        (cur_stage == 4'd1) ? W_INPROJ_Z_BASE :
        (cur_stage == 4'd4) ? W_XPROJ_BASE :
                              W_OUTPROJ_BASE;

    wire [14:0] m1a_wr_addr = PT_X_INNER_CIRC
                              + ({13'b0, t_cnt[1:0]} * {10'b0, ch_m_act})
                              + {10'b0, c_out_grp};
    wire [14:0] m1b_wr_addr = PT_Z_GATE + {10'b0, c_out_grp};
    wire [14:0] m4_wr_addr  = PT_X_PROJ + {10'b0, c_out_grp};
    wire [14:0] m8_wr_addr  = PT_MAMBA_OUT
                              + (t_cnt * {11'b0, CH_OUT})
                              + {10'b0, c_out_grp};
    wire [14:0] mac_wr_addr =
        (cur_stage == 4'd0) ? m1a_wr_addr :
        (cur_stage == 4'd1) ? m1b_wr_addr :
        (cur_stage == 4'd4) ? m4_wr_addr  :
                              m8_wr_addr;

    wire [8:0]  mac_last     = mac_len - 9'd1;
    wire [8:0]  mac_last2    = mac_len - 9'd2;
    wire [4:0]  mac_grp_last = mac_grp_count - 5'd1;
    wire [14:0] mac_len_ext  = {6'b0, mac_len};
    wire [14:0] w_grp_base   = mac_w_base + ({10'b0, c_out_grp} * mac_len_ext);

    wire [8:0]  c_in_p1 = c_in_cnt + 9'd1;
    wire [8:0]  c_in_p2 = c_in_cnt + 9'd2;
    wire [8:0]  c_in_p4 = c_in_cnt + 9'd4;

    wire [14:0] y_addr_now = mac_in_base + {11'b0, c_in_cnt[8:4]};
    wire [14:0] y_addr_p2  = mac_in_base + {11'b0, c_in_p2[8:4]};
    wire [14:0] y_addr_p4  = mac_in_base + {11'b0, c_in_p4[8:4]};
    wire [14:0] w_addr_now = w_grp_base + {6'b0, c_in_cnt};
    wire [14:0] w_addr_p2  = w_grp_base + {6'b0, c_in_p2};
    wire [14:0] w_addr_p4  = w_grp_base + {6'b0, c_in_p4};

    wire [4:0]  c_out_grp_next = c_out_grp + 5'd1;
    wire [14:0] w_grp_base_next = mac_w_base + ({10'b0, c_out_grp_next} * mac_len_ext);
    wire [14:0] w_addr_next_grp = w_grp_base_next;
    wire [14:0] y_addr_grp_start = mac_in_base;

    wire [4:0]  inner_grp_last = ch_m_act - 5'd1;

    // ============================================================
    //  RMSNorm helpers
    // ============================================================
    wire [4:0]  rn_x_per_t     = {1'b0, CH_OUT};
    wire [4:0]  rn_x_grp_last  = rn_x_per_t - 5'd1;
    wire [14:0] rn_x_per_t_ext = {10'b0, rn_x_per_t};
    wire [14:0] rn_t_offset_x  = t_cnt * rn_x_per_t_ext;
    wire [14:0] rn_x_addr_now  = PT_INPUT + rn_t_offset_x + {10'b0, rn_grp};
    wire [4:0]  rn_grp_p1      = rn_grp + 5'd1;
    wire [14:0] rn_g_addr_now  = C_W_NORM_BASE + {10'b0, rn_grp};
    wire [14:0] rn_wr_addr     = PT_X_NORM + {10'b0, rn_grp};
    wire [14:0] sq_x_addr_now  = PT_INPUT + rn_t_offset_x + {11'b0, c_in_cnt[3:0]};
    wire [14:0] sq_x_addr_p1   = PT_INPUT + rn_t_offset_x + {11'b0, c_in_p1[3:0]};
    wire [14:0] sq_x_addr_p2   = PT_INPUT + rn_t_offset_x + {11'b0, c_in_p2[3:0]};
    wire [8:0]  rn_sq_last     = {4'b0, rn_x_grp_last};

    // ============================================================
    //  M2 helpers (4-tap depth-wise conv, circular X_INNER_CIRC)
    // ============================================================
    wire signed [10:0] eff_t_for_tap = {1'b0, t_cnt} - 11'd3 + {9'b0, tap_cnt};
    wire        m2_pad     = (eff_t_for_tap < 0);
    wire [1:0]  eff_t_slot = eff_t_for_tap[1:0];
    wire [14:0] m2_x_addr_for_tap = PT_X_INNER_CIRC
                                    + ({13'b0, eff_t_slot} * {10'b0, ch_m_act})
                                    + {10'b0, c_out_grp};

    wire [2:0]  tap_p1 = {1'b0, tap_cnt} + 3'd1;
    wire [2:0]  tap_p2 = {1'b0, tap_cnt} + 3'd2;
    wire signed [11:0] eff_t_p1 = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p1});
    wire signed [11:0] eff_t_p2 = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p2});
    wire        m2_pad_p1 = (eff_t_p1 < 0);
    wire        m2_pad_p2 = (eff_t_p2 < 0);
    wire [1:0]  eff_t_slot_p1 = eff_t_p1[1:0];
    wire [1:0]  eff_t_slot_p2 = eff_t_p2[1:0];
    wire [14:0] m2_x_addr_p1 = PT_X_INNER_CIRC
                               + ({13'b0, eff_t_slot_p1} * {10'b0, ch_m_act})
                               + {10'b0, c_out_grp};
    wire [14:0] m2_x_addr_p2 = PT_X_INNER_CIRC
                               + ({13'b0, eff_t_slot_p2} * {10'b0, ch_m_act})
                               + {10'b0, c_out_grp};

    wire [14:0] m2_w_addr_for_tap = W_DW_BASE + ({9'b0, c_out_grp} * 15'd4) + {13'b0, tap_cnt};
    wire [14:0] m2_w_addr_p1      = W_DW_BASE + ({9'b0, c_out_grp} * 15'd4) + {12'b0, tap_p1};
    wire [14:0] m2_w_addr_p2      = W_DW_BASE + ({9'b0, c_out_grp} * 15'd4) + {12'b0, tap_p2};
    wire [14:0] m2_bias_addr      = C_B_DW_BASE + {10'b0, c_out_grp};
    wire [14:0] m2_wr_addr        = PT_X_CONV + {10'b0, c_out_grp};

    // ============================================================
    //  M3 helpers (SiLU elementwise)
    // ============================================================
    wire [14:0] m3_in_addr = PT_X_CONV + {10'b0, c_out_grp};
    wire [14:0] m3_wr_addr = PT_U      + {10'b0, c_out_grp};

    // ============================================================
    //  M5 helpers (dt_proj + softplus)
    // ============================================================
    wire [14:0]  m5_dt_addr      = PT_X_PROJ + 15'd0;
    wire [15:0]  m5_dt_scalar    = m5_dt_word_reg[m5_dt_lane*16 +: 16];
    wire [255:0] m5_dt_broadcast = {16{m5_dt_scalar}};
    wire [4:0]   dt_rank_last    = {1'b0, DT_RANK - 4'd1};
    wire [4:0]   m5_w_p1         = m5_w_cnt + 5'd1;
    wire [4:0]   m5_w_p2         = m5_w_cnt + 5'd2;
    wire [14:0]  m5_w_addr_now   = W_DTPROJ_BASE + ({10'b0, c_out_grp} * {11'b0, DT_RANK}) + {10'b0, m5_w_cnt};
    wire [14:0]  m5_w_addr_p1    = W_DTPROJ_BASE + ({10'b0, c_out_grp} * {11'b0, DT_RANK}) + {10'b0, m5_w_p1};
    wire [14:0]  m5_w_addr_p2    = W_DTPROJ_BASE + ({10'b0, c_out_grp} * {11'b0, DT_RANK}) + {10'b0, m5_w_p2};
    wire [14:0]  m5_bias_addr    = C_B_DT_BASE + {10'b0, c_out_grp};
    wire [14:0]  m5_wr_addr      = PT_DELTA + {10'b0, c_out_grp};

    // ============================================================
    //  M7 helpers (y_ssm * SiLU(z_gate))
    // ============================================================
    wire [14:0] m7_z_addr  = PT_Z_GATE + {10'b0, c_out_grp};
    wire [14:0] m7_y_addr  = PT_Y_SSM  + {10'b0, c_out_grp};
    wire [14:0] m7_wr_addr = PT_Y_GATED + {10'b0, c_out_grp};

    // ============================================================
    //  M6 helpers
    // ============================================================
    wire [15:0]  m6_dt_scalar    = m6_delta_word[m6_lane*16 +: 16];
    wire [255:0] m6_dt_broadcast = {16{m6_dt_scalar}};
    wire [15:0]  m6_u_scalar     = m6_u_word[m6_lane*16 +: 16];
    wire [255:0] m6_u_broadcast  = {16{m6_u_scalar}};
    wire [15:0]  m6_D_scalar     = m6_D_word[m6_lane*16 +: 16];
    wire [255:0] m6_D_broadcast  = {16{m6_D_scalar}};

    wire [8:0]   m6_c = {c_out_grp[3:0], m6_lane};

    wire [14:0]  m6_w0_addr      = PT_X_PROJ + 15'd0;
    wire [14:0]  m6_B_addr       = PT_X_PROJ + 15'd1;
    wire [14:0]  m6_C_addr       = PT_X_PROJ + 15'd2;
    wire [14:0]  m6_delta_addr   = PT_DELTA + {10'b0, c_out_grp};
    wire [14:0]  m6_u_addr       = PT_U     + {10'b0, c_out_grp};
    wire [14:0]  m6_D_addr_const = C_D_PARAM_BASE + {10'b0, c_out_grp};
    wire [14:0]  m6_A_addr_w     = W_A_BASE + {6'b0, m6_c};
    wire [14:0]  m6_y_ssm_wr_addr = PT_Y_SSM + {10'b0, c_out_grp};

    wire signed [`ACC_W+4:0] m6_y_ch_full = sum_d_wide >>> `FRAC_BITS;
    wire signed [15:0] m6_y_ch_sat =
        (m6_y_ch_full >  45'sd32767)  ? 16'sh7FFF :
        (m6_y_ch_full < -45'sd32768)  ? 16'sh8000 :
                                         m6_y_ch_full[15:0];
    wire signed [16:0] m6_ysum = {m6_y_ch_reg[15], m6_y_ch_reg} + {m6_du_reg[15], m6_du_reg};
    wire signed [15:0] m6_y_ssm_scalar =
        (m6_ysum >  17'sd32767)  ? 16'sh7FFF :
        (m6_ysum < -17'sd32768)  ? 16'sh8000 :
                                    m6_ysum[15:0];

    // ============================================================
    //  FSM
    // ============================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state              <= S_IDLE;
            cur_stage          <= 4'd0;
            t_cnt              <= 10'd0;
            c_out_grp          <= 5'd0;
            c_in_cnt           <= 9'd0;
            rn_grp             <= 5'd0;
            tap_cnt            <= 2'd0;
            m5_w_cnt           <= 5'd0;
            m5_dt_lane         <= 4'd0;
            m5_dt_word_reg     <= 256'd0;
            silu_z_reg         <= 256'd0;
            m_rd_addr          <= 15'd0;
            m_we               <= 1'b0;
            m_wr_addr          <= 15'd0;
            m_wr_data          <= 256'd0;
            w_rd_addr          <= 15'd0;
            w_rd_addr2         <= 15'd0;
            cl_op_mode         <= `MAMBA_PE_IDLE;
            cl_clear_acc       <= 1'b0;
            cl_in_W1_vec       <= 256'd0;
            cl_in_H_ext        <= 256'd0;
            cl_in_W2_vec       <= 256'd0;
            cl_in_X_vec        <= 256'd0;
            const_rd_addr_r    <= 15'd0;
            rsqrt_idx_r        <= 13'd0;
            S_reg              <= 16'd0;
            h_init_cnt         <= 9'd0;
            m6_lane            <= 4'd0;
            m6_delta_word      <= 256'd0;
            m6_u_word          <= 256'd0;
            m6_w0_reg          <= 256'd0;
            m6_w1_reg          <= 256'd0;
            m6_w2_reg          <= 256'd0;
            m6_D_word          <= 256'd0;
            m6_dA_reg          <= 256'd0;
            m6_dB_reg          <= 256'd0;
            m6_t1_reg          <= 256'd0;
            m6_t2_reg          <= 256'd0;
            m6_ssm_grp_acc     <= 256'd0;
            m6_y_ch_reg        <= 16'sd0;
            m6_du_reg          <= 16'sd0;
            m6_h_from_rf_r     <= 1'b0;
            m6_h_rd_addr_r     <= 9'd0;
            m6_h_wr_en_r       <= 1'b0;
            m6_h_wr_addr_r     <= 9'd0;
            m6_h_wr_from_pe_r  <= 1'b0;
            m6_h_wr_data_ext_r <= 256'd0;
            done_stage         <= 1'b0;
            done_all           <= 1'b0;
        end else begin
            cl_op_mode   <= `MAMBA_PE_IDLE;
            cl_clear_acc <= 1'b0;
            m_we         <= 1'b0;
            m6_h_wr_en_r <= 1'b0;

            case (state)
                S_IDLE: begin
                    done_stage <= 1'b0;
                    done_all   <= 1'b0;
                    if (start) begin
                        t_cnt      <= 10'd0;
                        c_out_grp  <= 5'd0;
                        c_in_cnt   <= 9'd0;
                        rn_grp     <= 5'd0;
                        tap_cnt    <= 2'd0;
                        m5_w_cnt   <= 5'd0;
                        m5_dt_lane <= 4'd0;
                        m6_lane    <= 4'd0;
                        h_init_cnt <= 9'd0;
                        cur_stage  <= 4'd6;   // during H_INIT, mark M6 to route h_ctrl
                        state      <= S_H_INIT;
                    end
                end

                // ============== H_RegFile zero-init (once) ==============
                // h_init_cnt runs 0..d_inner; issue write on cycles 0..d_inner-1,
                // then transition on cycle d_inner (last write completes safely
                // because h_ctrl_active stays true while state == S_H_INIT).
                S_H_INIT: begin
                    if (h_init_cnt < d_inner) begin
                        m6_h_wr_en_r       <= 1'b1;
                        m6_h_wr_addr_r     <= h_init_cnt;
                        m6_h_wr_from_pe_r  <= 1'b0;
                        m6_h_wr_data_ext_r <= 256'd0;
                        h_init_cnt         <= h_init_cnt + 9'd1;
                    end else begin
                        h_init_cnt <= 9'd0;
                        cur_stage  <= 4'd9;
                        state      <= S_RN_SQ_PREF;
                    end
                end

                // ============== MAC stages (M1A/M1B/M4/M8) ==============
                S_PREFETCH: begin
                    m_rd_addr  <= y_addr_now;
                    w_rd_addr  <= w_addr_now;
                    w_rd_addr2 <= w_addr_now + 15'd1;
                    state      <= S_WAIT;
                end
                S_WAIT: begin
                    w_rd_addr    <= w_addr_p2;
                    w_rd_addr2   <= w_addr_p2 + 15'd1;
                    m_rd_addr    <= y_addr_p2;
                    c_in_cnt     <= 9'd0;
                    // PE stays IDLE; MAC2 fires from S_MAC_LOAD onwards.
                    state        <= S_MAC_LOAD;
                end
                S_MAC_LOAD: begin
                    // Capture pair 0 (w[0..1], y[0..1]) into PE input regs.
                    // BRAM/URAM data is valid this cycle from S_WAIT's addr.
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_W2_vec <= w_rd_data2;
                    cl_in_H_ext  <= y_broadcast;
                    cl_in_X_vec  <= y_broadcast2;
                    // Prepare PE for first MAC on next edge.
                    cl_op_mode   <= `MAMBA_PE_MAC2;
                    cl_clear_acc <= 1'b1;
                    // Advance addresses one pair ahead.
                    m_rd_addr    <= y_addr_p4;
                    w_rd_addr    <= w_addr_p4;
                    w_rd_addr2   <= w_addr_p4 + 15'd1;
                    c_in_cnt     <= c_in_p2;
                    state        <= S_MAC;
                end
                S_MAC: begin
                    cl_op_mode <= `MAMBA_PE_MAC2;
                    // Capture next pair every cycle; PE consumes previous pair.
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_W2_vec <= w_rd_data2;
                    cl_in_H_ext  <= y_broadcast;
                    cl_in_X_vec  <= y_broadcast2;
                    m_rd_addr  <= y_addr_p4;
                    w_rd_addr  <= w_addr_p4;
                    w_rd_addr2 <= w_addr_p4 + 15'd1;
                    c_in_cnt   <= c_in_p2;
                    // Exit when PE consumes the last pair (c_in_cnt == mac_len).
                    // Register write from cl_out_vec breaks the DSP→sat→wr chain.
                    if (c_in_cnt == mac_last2 + 9'd2) begin
                        state <= S_MAC_LATCH;
                    end
                end
                S_MAC_LATCH: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= mac_wr_addr;
                    m_wr_data <= cl_out_vec;   // registered path
                    state     <= S_NEXT;
                end
                S_NEXT: begin
                    if (c_out_grp == mac_grp_last) begin
                        c_out_grp <= 5'd0;
                        c_in_cnt  <= 9'd0;
                        // Chain to next stage
                        case (cur_stage)
                            4'd0: begin cur_stage <= 4'd1; state <= S_PREFETCH; end
                            4'd1: begin cur_stage <= 4'd2; state <= S_M2_PREF;  end
                            4'd4: begin cur_stage <= 4'd5; state <= S_M5_DT_READ; end
                            4'd8: begin
                                if (t_cnt == t_last) begin
                                    state <= S_DONE;
                                end else begin
                                    t_cnt     <= t_cnt + 10'd1;
                                    cur_stage <= 4'd9;
                                    state     <= S_RN_SQ_PREF;
                                end
                            end
                            default: state <= S_IDLE;
                        endcase
                    end else begin
                        c_out_grp  <= c_out_grp_next;
                        c_in_cnt   <= 9'd0;
                        m_rd_addr  <= y_addr_grp_start;
                        w_rd_addr  <= w_addr_next_grp;
                        w_rd_addr2 <= w_addr_next_grp + 15'd1;
                        state      <= S_WAIT;
                    end
                end

                // ============== RMSNorm ==============
                S_RN_SQ_PREF: begin
                    c_in_cnt  <= 9'd0;
                    m_rd_addr <= sq_x_addr_now;
                    state     <= S_RN_SQ_WAIT;
                end
                S_RN_SQ_WAIT: begin
                    m_rd_addr <= sq_x_addr_p1;
                    state     <= S_RN_SQ_MAC;
                end
                S_RN_SQ_MAC: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (c_in_cnt == 9'd0);
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= m_rd_data;
                    m_rd_addr    <= sq_x_addr_p2;
                    c_in_cnt     <= c_in_p1;
                    if (c_in_cnt == rn_sq_last) state <= S_RN_SQ_FINAL;
                end
                S_RN_SQ_FINAL: state <= S_RN_SQ_DONE;
                S_RN_SQ_DONE: begin
                    rsqrt_idx_r <= mean_i_clip;
                    state       <= S_RN_RSQ_WAIT;
                end
                S_RN_RSQ_WAIT: state <= S_RN_S_LATCH;
                S_RN_S_LATCH: begin
                    S_reg  <= rsqrt_data;
                    rn_grp <= 5'd0;
                    state  <= S_RN_AP_PREF;
                end
                S_RN_AP_PREF: begin
                    m_rd_addr       <= rn_x_addr_now;
                    const_rd_addr_r <= rn_g_addr_now;
                    state           <= S_RN_AP_WAIT;
                end
                S_RN_AP_WAIT: state <= S_RN_AP_MUL1;
                S_RN_AP_MUL1: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= const_rd_data;
                    state        <= S_RN_AP_WAIT1;
                end
                S_RN_AP_WAIT1: state <= S_RN_AP_MUL2;
                S_RN_AP_MUL2: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= cl_out_vec;
                    cl_in_H_ext  <= S_broadcast;
                    state        <= S_RN_AP_WAIT2;
                end
                S_RN_AP_WAIT2: state <= S_RN_AP_WRITE;
                S_RN_AP_WRITE: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= rn_wr_addr;
                    m_wr_data <= cl_out_vec;
                    state     <= S_RN_AP_NEXT;
                end
                S_RN_AP_NEXT: begin
                    if (rn_grp == rn_x_grp_last) begin
                        rn_grp    <= 5'd0;
                        c_out_grp <= 5'd0;
                        c_in_cnt  <= 9'd0;
                        cur_stage <= 4'd0;
                        state     <= S_PREFETCH;   // → M1A
                    end else begin
                        rn_grp <= rn_grp + 5'd1;
                        state  <= S_RN_AP_PREF;
                    end
                end

                // ============== M2 depth-wise conv + bias ==============
                S_M2_PREF: begin
                    tap_cnt   <= 2'd0;
                    m_rd_addr <= m2_x_addr_for_tap;
                    w_rd_addr <= m2_w_addr_for_tap;
                    state     <= S_M2_WAIT;
                end
                S_M2_WAIT: begin
                    m_rd_addr <= m2_x_addr_p1;
                    w_rd_addr <= m2_w_addr_p1;
                    state     <= S_M2_TAP;
                end
                S_M2_TAP: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (tap_cnt == 2'd0);
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_H_ext  <= m2_pad ? 256'b0 : m_rd_data;
                    m_rd_addr    <= m2_x_addr_p2;
                    w_rd_addr    <= m2_w_addr_p2;
                    tap_cnt      <= tap_p1[1:0];
                    if (tap_cnt == 2'd3) state <= S_M2_FINAL;
                end
                S_M2_FINAL: state <= S_M2_BIAS_PREF;
                S_M2_BIAS_PREF: begin
                    cl_in_W1_vec    <= cl_out_vec;
                    const_rd_addr_r <= m2_bias_addr;
                    state           <= S_M2_BIAS_WAIT;
                end
                S_M2_BIAS_WAIT: state <= S_M2_BIAS_ADD;
                S_M2_BIAS_ADD: begin
                    cl_op_mode  <= `MAMBA_PE_ADD;
                    cl_in_H_ext <= const_rd_data;
                    state       <= S_M2_BIAS_LATCH;
                end
                S_M2_BIAS_LATCH: state <= S_M2_WRITE;
                S_M2_WRITE: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m2_wr_addr;
                    m_wr_data <= cl_out_vec;
                    state     <= S_M2_NEXT;
                end
                S_M2_NEXT: begin
                    if (c_out_grp == inner_grp_last) begin
                        c_out_grp <= 5'd0;
                        cur_stage <= 4'd3;
                        state     <= S_M3_PREF;
                    end else begin
                        c_out_grp <= c_out_grp + 5'd1;
                        state     <= S_M2_PREF;
                    end
                end

                // ============== M3 SiLU ==============
                S_M3_PREF: begin
                    m_rd_addr <= m3_in_addr;
                    state     <= S_M3_WAIT;
                end
                S_M3_WAIT: state <= S_M3_WRITE;
                S_M3_WRITE: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m3_wr_addr;
                    m_wr_data <= silu_out_w;
                    state     <= S_M3_NEXT;
                end
                S_M3_NEXT: begin
                    if (c_out_grp == inner_grp_last) begin
                        c_out_grp <= 5'd0;
                        c_in_cnt  <= 9'd0;
                        cur_stage <= 4'd4;
                        state     <= S_PREFETCH;   // → M4
                    end else begin
                        c_out_grp <= c_out_grp + 5'd1;
                        state     <= S_M3_PREF;
                    end
                end

                // ============== M5 dt_proj + bias + softplus ==============
                S_M5_DT_READ: begin
                    m_rd_addr <= m5_dt_addr;
                    state     <= S_M5_DT_WAIT;
                end
                S_M5_DT_WAIT: state <= S_M5_DT_LATCH;
                S_M5_DT_LATCH: begin
                    m5_dt_word_reg <= m_rd_data;
                    m5_dt_lane     <= 4'd0;
                    m5_w_cnt       <= 5'd0;
                    state          <= S_M5_W_PREF;
                end
                S_M5_W_PREF: begin
                    w_rd_addr <= m5_w_addr_now;
                    state     <= S_M5_W_WAIT;
                end
                S_M5_W_WAIT: begin
                    w_rd_addr <= m5_w_addr_p1;
                    state     <= S_M5_MAC;
                end
                S_M5_MAC: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (m5_w_cnt == 5'd0);
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_H_ext  <= m5_dt_broadcast;
                    w_rd_addr    <= m5_w_addr_p2;
                    m5_dt_lane   <= m5_dt_lane + 4'd1;
                    m5_w_cnt     <= m5_w_p1;
                    if (m5_w_cnt == dt_rank_last) state <= S_M5_FINAL;
                end
                S_M5_FINAL: state <= S_M5_BIAS_PREF;
                S_M5_BIAS_PREF: begin
                    cl_in_W1_vec    <= cl_out_vec;
                    const_rd_addr_r <= m5_bias_addr;
                    state           <= S_M5_BIAS_WAIT;
                end
                S_M5_BIAS_WAIT: state <= S_M5_BIAS_ADD;
                S_M5_BIAS_ADD: begin
                    cl_op_mode  <= `MAMBA_PE_ADD;
                    cl_in_H_ext <= const_rd_data;
                    state       <= S_M5_BIAS_LATCH;
                end
                S_M5_BIAS_LATCH: state <= S_M5_WRITE;
                S_M5_WRITE: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m5_wr_addr;
                    m_wr_data <= sp_out_w;
                    state     <= S_M5_NEXT;
                end
                S_M5_NEXT: begin
                    m5_dt_lane <= 4'd0;
                    m5_w_cnt   <= 5'd0;
                    if (c_out_grp == inner_grp_last) begin
                        c_out_grp <= 5'd0;
                        cur_stage <= 4'd6;
                        state     <= S_M6_LOAD_W0_PREF;
                    end else begin
                        c_out_grp <= c_out_grp + 5'd1;
                        state     <= S_M5_W_PREF;
                    end
                end

                // ============== M6 SSM scan ==============
                // Load 3 raw X_PROJ words (once per timestep) to reconstruct
                // aligned B/C via combinational shuffle wires m6_B_word/m6_C_word.
                S_M6_LOAD_W0_PREF: begin
                    m_rd_addr <= m6_w0_addr;
                    state     <= S_M6_LOAD_W0_WAIT;
                end
                S_M6_LOAD_W0_WAIT: state <= S_M6_LOAD_W0_LATCH;
                S_M6_LOAD_W0_LATCH: begin
                    m6_w0_reg <= m_rd_data;
                    state     <= S_M6_LOAD_B_PREF;
                end
                S_M6_LOAD_B_PREF: begin
                    m_rd_addr <= m6_B_addr;
                    state     <= S_M6_LOAD_B_WAIT;
                end
                S_M6_LOAD_B_WAIT: state <= S_M6_LOAD_B_LATCH;
                S_M6_LOAD_B_LATCH: begin
                    m6_w1_reg <= m_rd_data;
                    state     <= S_M6_LOAD_C_PREF;
                end
                S_M6_LOAD_C_PREF: begin
                    m_rd_addr <= m6_C_addr;
                    state     <= S_M6_LOAD_C_WAIT;
                end
                S_M6_LOAD_C_WAIT: state <= S_M6_LOAD_C_LATCH;
                S_M6_LOAD_C_LATCH: begin
                    m6_w2_reg <= m_rd_data;
                    c_out_grp <= 5'd0;
                    state     <= S_M6_LOAD_DT_PREF;
                end
                S_M6_LOAD_DT_PREF: begin
                    m_rd_addr <= m6_delta_addr;
                    state     <= S_M6_LOAD_DT_WAIT;
                end
                S_M6_LOAD_DT_WAIT: state <= S_M6_LOAD_DT_LATCH;
                S_M6_LOAD_DT_LATCH: begin
                    m6_delta_word <= m_rd_data;
                    state         <= S_M6_LOAD_U_PREF;
                end
                S_M6_LOAD_U_PREF: begin
                    m_rd_addr <= m6_u_addr;
                    state     <= S_M6_LOAD_U_WAIT;
                end
                S_M6_LOAD_U_WAIT: state <= S_M6_LOAD_U_LATCH;
                S_M6_LOAD_U_LATCH: begin
                    m6_u_word <= m_rd_data;
                    state     <= S_M6_LOAD_D_PREF;
                end
                S_M6_LOAD_D_PREF: begin
                    const_rd_addr_r <= m6_D_addr_const;
                    state           <= S_M6_LOAD_D_WAIT;
                end
                S_M6_LOAD_D_WAIT: state <= S_M6_LOAD_D_LATCH;
                S_M6_LOAD_D_LATCH: begin
                    m6_D_word <= const_rd_data;
                    m6_lane   <= 4'd0;
                    state     <= S_M6_A_PREF;
                end

                S_M6_A_PREF: begin
                    w_rd_addr <= m6_A_addr_w;
                    state     <= S_M6_A_WAIT;
                end
                S_M6_A_WAIT: state <= S_M6_DAB_MUL2;
                // MUL2 fusion: one PE op computes both m1 = dt*A and m2 = dt*B.
                //   W1=W2=dt_broadcast, H=A (w_rd_data), X=B (m6_B_word)
                //   PE registers cl_out_vec ← sat(m1>>FB), cl_out_vec2 ← sat(m2>>FB).
                S_M6_DAB_MUL2: begin
                    cl_op_mode   <= `MAMBA_PE_MUL2;
                    cl_in_W1_vec <= m6_dt_broadcast;
                    cl_in_H_ext  <= w_rd_data;
                    cl_in_W2_vec <= m6_dt_broadcast;
                    cl_in_X_vec  <= m6_B_word;
                    state        <= S_M6_EXP_WAIT;
                end
                // Wait cycle: PE now has MUL2 inputs registered; cl_out_vec /
                // cl_out_vec2 update on the next edge.
                S_M6_EXP_WAIT: state <= S_M6_DAB_LATCH;
                // Both dA (via Exp_LUT of registered cl_out_vec) and dB
                // (direct registered cl_out_vec2) captured in one state.
                S_M6_DAB_LATCH: begin
                    m6_dA_reg <= exp_out_w;
                    m6_dB_reg <= cl_out_vec2;
                    state     <= S_M6_H_PREF;
                end
                S_M6_H_PREF: begin
                    m6_h_from_rf_r <= 1'b1;
                    m6_h_rd_addr_r <= m6_c[8:0];
                    state          <= S_M6_H_WAIT;
                end
                S_M6_H_WAIT: state <= S_M6_T1_MUL;
                S_M6_T1_MUL: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m6_dA_reg;
                    state        <= S_M6_T1_WAIT;
                end
                // Wait 1 cycle for PE to register cl_out_vec ← dA·h.
                S_M6_T1_WAIT: state <= S_M6_T1_LATCH;
                S_M6_T1_LATCH: begin
                    m6_t1_reg      <= cl_out_vec;    // registered path
                    m6_h_from_rf_r <= 1'b0;
                    state          <= S_M6_T2_MUL;
                end
                S_M6_T2_MUL: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m6_dB_reg;
                    cl_in_H_ext  <= m6_u_broadcast;
                    state        <= S_M6_T2_WAIT;
                end
                S_M6_T2_WAIT: state <= S_M6_T2_LATCH;
                S_M6_T2_LATCH: begin
                    m6_t2_reg <= cl_out_vec;         // registered path
                    state     <= S_M6_H_ADD;
                end
                S_M6_H_ADD: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= m6_t1_reg;
                    cl_in_H_ext  <= m6_t2_reg;
                    state        <= S_M6_H_WRITE;
                end
                S_M6_H_WRITE: begin
                    m6_h_wr_en_r      <= 1'b1;
                    m6_h_wr_addr_r    <= m6_c[8:0];
                    m6_h_wr_from_pe_r <= 1'b1;
                    state             <= S_M6_Y_MAC;
                end
                S_M6_Y_MAC: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= 1'b1;
                    cl_in_W1_vec <= cl_out_vec;
                    cl_in_H_ext  <= m6_C_word;
                    state        <= S_M6_Y_WAIT;
                end
                S_M6_Y_WAIT: state <= S_M6_Y_LATCH;
                S_M6_Y_LATCH: begin
                    m6_y_ch_reg <= m6_y_ch_sat;
                    state       <= S_M6_DU_MUL;
                end
                S_M6_DU_MUL: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m6_D_broadcast;
                    cl_in_H_ext  <= m6_u_broadcast;
                    state        <= S_M6_DU_WAIT;
                end
                S_M6_DU_WAIT: state <= S_M6_DU_LATCH;
                S_M6_DU_LATCH: begin
                    m6_du_reg <= cl_out_vec[15:0];    // registered path
                    state     <= S_M6_FINALIZE;
                end
                S_M6_FINALIZE: begin
                    m6_ssm_grp_acc[m6_lane*16 +: 16] <= m6_y_ssm_scalar;
                    state <= S_M6_LANE_NEXT;
                end
                S_M6_LANE_NEXT: begin
                    if (m6_lane == 4'd15) begin
                        state <= S_M6_WRITE_GRP;
                    end else begin
                        m6_lane <= m6_lane + 4'd1;
                        state   <= S_M6_A_PREF;
                    end
                end
                S_M6_WRITE_GRP: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m6_y_ssm_wr_addr;
                    m_wr_data <= m6_ssm_grp_acc;
                    state     <= S_M6_GRP_NEXT;
                end
                S_M6_GRP_NEXT: begin
                    if (c_out_grp == inner_grp_last) begin
                        c_out_grp <= 5'd0;
                        cur_stage <= 4'd7;
                        state     <= S_M7_Z_PREF;
                    end else begin
                        c_out_grp <= c_out_grp + 5'd1;
                        m6_lane   <= 4'd0;
                        state     <= S_M6_LOAD_DT_PREF;
                    end
                end

                // ============== M7 gating ==============
                S_M7_Z_PREF: begin
                    m_rd_addr <= m7_z_addr;
                    state     <= S_M7_Z_WAIT;
                end
                S_M7_Z_WAIT: state <= S_M7_Z_LATCH;
                S_M7_Z_LATCH: begin
                    silu_z_reg <= silu_out_w;
                    state      <= S_M7_Y_PREF;
                end
                S_M7_Y_PREF: begin
                    m_rd_addr <= m7_y_addr;
                    state     <= S_M7_Y_WAIT;
                end
                S_M7_Y_WAIT: state <= S_M7_MUL;
                S_M7_MUL: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= silu_z_reg;
                    state        <= S_M7_LATCH;
                end
                S_M7_LATCH: state <= S_M7_WRITE;
                S_M7_WRITE: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m7_wr_addr;
                    m_wr_data <= cl_out_vec;
                    state     <= S_M7_NEXT;
                end
                S_M7_NEXT: begin
                    if (c_out_grp == inner_grp_last) begin
                        c_out_grp <= 5'd0;
                        c_in_cnt  <= 9'd0;
                        cur_stage <= 4'd8;
                        state     <= S_PREFETCH;   // → M8
                    end else begin
                        c_out_grp <= c_out_grp + 5'd1;
                        state     <= S_M7_Z_PREF;
                    end
                end

                S_DONE: begin
                    done_stage <= 1'b1;
                    done_all   <= 1'b1;
                    if (!start) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
