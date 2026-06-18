`include "_parameter.v"
`include "_block_params.v"

// ============================================================================
// ITM_Top_v3 — D2 end-to-end fork of ITM_Top (v1 base).
//
// Adds three new phases integrated into the same FSM, sharing PE_Array,
// Memory_System, Const_Storage, and existing sat_add16 / bn_relu helpers:
//
//   1. ENCODER (enc_mode=1, before block 0)
//        Mathematically identical to P1 but with d_in=12 (zero-padded to 16),
//        d_out=64.  Reads raw 12-lead waveform from B_ENC_IN_BASE in ram_b,
//        loads encoder weights from W_ENC_BASE in ram_weight and encoder
//        bias from C_ENC_BIAS in ram_const, writes encoder output to
//        A_INPUT_BASE in ram_a (block 0 reads from there naturally).
//
//   2. GAP (head_mode=1, after last block FIN when cascade_mode=0)
//        Per-channel mean over T_GAP using gap_q = sat16((sum * INV_T_Q15) >> 15).
//        Reads FINAL_OUT (split across A_FINAL_OUT / B_FINAL_OUT per c_grp).
//        Stores 8 c_grp × 256-bit GAP output to internal gap_q_reg[0..7].
//
//   3. FC (after GAP)
//        Linear(128 → 5).  MAC reduction × 5 classes using PE_Array MODE_MAC.
//        Reads FC weights from W_FC_BASE, FC bias from C_FC_BIAS, GAP input
//        from gap_q_reg.  Drives 5 logits to top-level logit0..4 registers.
//
// Both new modes off → identical behavior to v1.  enc_mode and head_mode are
// independent (use only encoder, or only head, or both, depending on host
// orchestration).
//
// Supports all 5 ITM block configs (same as v1):
//   block 0,1: T=1000, CH_IN=4 (d_in=64),  CH_M=8  (d_inner=128), DT_RANK=4
//   block 2,3: T=500,  CH_IN=4,             CH_M=8,                DT_RANK=4
//   block 4:   T=250,  CH_IN=4 (d_in=64), CH_OUT=8 (d_out=128), CH_M=16, DT_RANK=8
// ============================================================================

module ITM_Top_v3 (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         done_phase1,
    output reg         done_inception,
    output reg         done_mamba,
    output reg         done_all,

    // ---- D2 end-to-end status ----
    output reg         done_encoder,
    output reg         done_gap,
    output reg         done_fc,

    input  wire [9:0]  T_MAX,
    input  wire [3:0]  CH_IN,
    input  wire [3:0]  CH_OUT,
    input  wire [3:0]  CH_M,
    input  wire [3:0]  DT_RANK,

    // Cascade control (sampled at start):
    //   need_pool=1     : after FIN, run stride-2 MaxPool over T into A_INPUT_BASE
    //   cascade_mode=1  : after FIN (and pool if any), copy/pool result into A_INPUT_BASE
    //                     for next block's P1 to consume. When 0, FINAL_OUT is the
    //                     terminal output (last block / host read-back).
    input  wire        need_pool,
    input  wire        cascade_mode,

    // ---- D2 phase enables (latched at start) ----
    input  wire        enc_mode,        // 1 = run encoder before P1
    input  wire        head_mode,       // 1 = run GAP + FC after last block FIN
    input  wire [9:0]  T_ENC,           // encoder timesteps (typically 1000)
    input  wire [9:0]  T_GAP,           // GAP timesteps    (typically 250)

    // ---- D2 FC output (5 logits, latched after done_fc) ----
    output reg signed [15:0] logit0,
    output reg signed [15:0] logit1,
    output reg signed [15:0] logit2,
    output reg signed [15:0] logit3,
    output reg signed [15:0] logit4,

    input  wire        dma_write_en,
    input  wire [1:0]  dma_target,
    input  wire [14:0] dma_addr,
    input  wire [255:0] dma_wdata,
    output wire        dma_ready,
    // ---- DMA READ interface: host reads back BRAM contents ----
    //   dma_rtarget = 0/1/2/3  ->  ram_a / ram_b / ram_weight / ram_const
    //   dma_rdata is registered (1-cycle latency from dma_raddr presented).
    input  wire        dma_read_en,
    input  wire [1:0]  dma_rtarget,
    input  wire [14:0] dma_raddr,
    output wire [255:0] dma_rdata
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
    wire [8:0]  d_inner      = {ch_m_actual[4:0], 4'b0};   // 9-bit: handles 256 for block4 (CH_M=16)
    wire [7:0]  d_inner_last = d_inner[7:0] - 8'd1;        // 8-bit: 256→255, 128→127

    wire [14:0] w_dw_size    = {7'd0, ch_m_actual} * 15'd4;               // CH_M * 4
    wire [14:0] W_XPROJ_BASE_W = W_M_DW_BASE + w_dw_size;
    wire [14:0] xproj_sz    = 15'd3 * {6'd0, d_inner};                   // 3 * d_inner
    wire [14:0] W_DTPROJ_BASE  = W_XPROJ_BASE_W + xproj_sz;
    wire [14:0] dtproj_sz   = {7'd0, ch_m_actual} * {11'd0, DT_RANK};    // CH_M * DT_RANK
    wire [14:0] W_ALOG_BASE    = W_DTPROJ_BASE + dtproj_sz;
    wire [14:0] alog_sz     = {7'd0, ch_m_actual} * 15'd16;              // CH_M * d_state(16)
    wire [14:0] W_DPARAM_BASE  = W_ALOG_BASE + alog_sz;
    wire [14:0] dparam_sz   = {7'd0, ch_m_actual};                       // CH_M
    wire [14:0] W_OUTPROJ_BASE = W_DPARAM_BASE + dparam_sz;

    // CP-4 fix: t_stride_* are now REGISTERED incremental accumulators, updated in
    // lockstep with t_cnt via tasks t_cnt_zero / t_cnt_inc (see below). This removes
    // four combinational multipliers (t_cnt * CH_IN/CH_OUT/ch_m_actual/3) from every
    // address path. NBA updates inside always @(posedge clk) keep strides perfectly
    // in sync with t_cnt — both registers latch at the same clock edge.
    reg  [14:0] t_stride_in;
    reg  [14:0] t_stride_m;
    reg  [14:0] t_stride_out;
    reg  [14:0] t_stride_xp;

    // ----------------------------------------------------------------------
    // Compact memory map (RAM-2): aggressive overlap of temporal-disjoint
    // regions.  See RAM_LAYOUT_PLAN.md for lifetime analysis.
    //
    // Bank A (ram_a) — peak 17256 words:
    //   [0,8000)      A_INPUT_BASE | A_X_INNER | A_BOT_OUT
    //                 INPUT alive P1 only; X_INNER alive M1A..M6A;
    //                 BOT_OUT alive during BR only.  All sequential, share base.
    //   [8000,16000)  A_Z_GATE | A_MAMBA_OUT | A_FINAL_OUT
    //                 Z_GATE alive M1B..M7; MAMBA_OUT M8..FIN ([8000,12000));
    //                 FINAL_OUT FIN..end ([12000,16000)).  Z_GATE dead before
    //                 M8 → MAMBA_OUT writes can start.  MAMBA_OUT and
    //                 FINAL_OUT alive concurrently in FIN but at disjoint
    //                 addresses (8000-12000 vs 12000-16000).
    //   [16000,17000) A_CH1_OUT — alive BR..FIN, dedicated slot
    //   [17000,17256) A_H_STATE — alive M6 only, dedicated slot
    //
    // Bank B (ram_b) — peak 19000 words:
    //   [0,8000)      B_P1_OUT | B_X_CONV | B_FINAL_OUT
    //                 Sequential lifetimes: P1_OUT (P1..M1B) → X_CONV (M2..M5)
    //                 → Y_SSM (M6B..M7) → FINAL_OUT (FIN..end).  Share base.
    //   [8000,16000)  B_U_SAFE | B_Y_SSM — alive M3CP..M6B
    //   [16000,17000) B_CH2_OUT — alive BR..FIN
    //   [17000,18000) B_CH3_OUT — alive BR..FIN
    //   [18000,19000) B_CH4_OUT — alive BR..FIN
    // ----------------------------------------------------------------------
    localparam A_INPUT_BASE = 15'd0;
    localparam A_X_INNER    = 15'd0;       // overlaps A_INPUT_BASE (temporal disjoint)
    localparam A_BOT_OUT    = 15'd0;       // overlaps A_INPUT_BASE / A_X_INNER (temporal disjoint)
    localparam A_Z_GATE     = 15'd8000;
    localparam A_MAMBA_OUT  = 15'd8000;    // overlaps A_Z_GATE first half (temporal disjoint)
    localparam A_FINAL_OUT  = 15'd12000;   // overlaps A_Z_GATE second half (temporal disjoint)
    localparam A_CH1_OUT    = 15'd16000;
    localparam A_H_STATE    = 15'd17000;

    localparam B_P1_OUT     = 15'd0;
    localparam B_X_CONV     = 15'd0;       // overlaps B_P1_OUT (temporal disjoint)
    localparam B_Y_SSM      = 15'd8000;    // overlaps B_P1_OUT / B_X_CONV (temporal disjoint)
    localparam B_FINAL_OUT  = 15'd0;       // overlaps all above (temporal disjoint)
    localparam B_U_SAFE     = 15'd8000;
    localparam B_CH2_OUT    = 15'd16000;
    localparam B_CH3_OUT    = 15'd17000;
    localparam B_CH4_OUT    = 15'd18000;
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
    // Const RAM layout sized for block 4 (max CH_OUT=8, max CH_M=16) to avoid overlaps.
    localparam C_P1_BIAS    = 15'd0;     // size CH_OUT (<=8)
    localparam C_INC_SCALE  = 15'd8;     // size CH_OUT (<=8)
    localparam C_INC_SHIFT  = 15'd16;    // size CH_OUT (<=8)
    localparam C_M_DW_BIAS  = 15'd24;    // size CH_M (<=16)
    localparam C_M_DT_BIAS  = 15'd40;    // size CH_M (<=16)
    localparam C_NORM_W     = 15'd56;    // size CH_OUT (<=8) - RMSNorm gamma weights

    // --------------------------------------------------------------
    // D2 end-to-end memory map additions
    //
    // ram_b free tail [19000, 20480) → 1480 words, holds 1000-step raw waveform.
    // ram_weight free tail [14000, 16384) → 2384 words, holds encoder + FC weights.
    // ram_const expanded to 128 entries → holds encoder + FC bias.
    // --------------------------------------------------------------
    localparam B_ENC_IN_BASE = 15'd19000;   // raw 12-lead waveform in ram_b
                                            // 1 word/timestep; lanes [0..11] valid, [12..15] = 0
    localparam W_ENC_BASE    = 15'd14000;   // encoder weight in ram_weight
                                            // 64 words = 64 output channels, lanes [0..11] valid
    localparam W_FC_BASE     = 15'd14064;   // FC weight in ram_weight: 5 classes × 8 word/class = 40 words
                                            // Word layout: class_c × 8 + grp_in (grp_in = 0..7, 16 in-channels/word)
    localparam C_ENC_BIAS    = 15'd64;      // encoder bias: 4 words (64 channels / 16 lanes)
    localparam C_FC_BIAS     = 15'd68;      // FC bias: 1 word (5 lanes used, lanes 5..15 = 0)

    // GAP reciprocal-T constant.  For T_GAP=250: round(2^15 / 250) = 131.
    // Matches Python pipeline hw_gap: gap_q = sat16((sum_q * INV_T_Q15) >> 15).
    localparam signed [15:0] INV_T_Q15 = 16'sd131;

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

    // Memory_System holds bulk R/W working memory (ram_a/b + ram_weight).
    // Constants (ram_const + activation LUTs + rsqrt ROM) live in Const_Storage,
    // instantiated further below.  DMA signals fan out to both modules; top-level
    // dma_rdata is muxed between them based on dma_rtarget==2'd3.
    wire [255:0] mem_dma_rdata;
    wire [255:0] const_dma_rdata;
    Memory_System mem_sys (
        .clk(clk), .reset(rst), .bank_sel(bank_sel),
        .core_read_addr(m_rd_addr),   .core_read_data(m_rd_data),
        .core_write_en(m_we),         .core_write_addr(m_wr_addr), .core_write_data(m_wr_data),
        .weight_read_addr(w_rd_addr), .weight_read_data(w_rd_data),
        .dma_write_en(dma_write_en),  .dma_target(dma_target),
        .dma_addr(dma_addr),          .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en),    .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr),        .dma_rdata(mem_dma_rdata)
    );
    assign dma_rdata = (dma_rtarget == 2'd3) ? const_dma_rdata : mem_dma_rdata;

    PE_Array pe_arr (
        .clk(clk), .rst(rst),
        .clear_acc(pe_clear),   .op_mode(pe_op_mode),
        .in_A(pe_A),            .in_A_vec(pe_A_vec), .a_is_vector(pe_a_is_vector),
        .in_B(pe_B),
        .out_vector(pe_out)
    );

    // ----------------------------------------------------------------------
    // Activation LUT lanes (16 each: silu, softplus, exp).  Arrays are wired
    // from FSM data sources (m_rd_data / dt_lane / pe_out) on the input side
    // and feed write-back muxes on the output side.  The actual tables live
    // in Const_Storage; see D3.A refactor in OPTIMIZATION_NOTES.md.
    // ----------------------------------------------------------------------
    wire signed [15:0] silu_in [0:15], silu_o [0:15];
    wire signed [15:0] sp_in   [0:15], sp_o   [0:15];
    wire signed [15:0] exp_in  [0:15], exp_o  [0:15];

    // Pack/unpack array ↔ 256-bit flat for Const_Storage's interface.
    wire [255:0] silu_in_flat,  sp_in_flat,  exp_in_flat;
    wire [255:0] silu_out_flat, sp_out_flat, exp_out_flat;
    genvar gp;
    generate
        for (gp = 0; gp < 16; gp = gp + 1) begin : PACK_UNPACK
            assign silu_in_flat[gp*16 +: 16] = silu_in[gp];
            assign sp_in_flat  [gp*16 +: 16] = sp_in  [gp];
            assign exp_in_flat [gp*16 +: 16] = exp_in [gp];
            assign silu_o[gp] = silu_out_flat[gp*16 +: 16];
            assign sp_o  [gp] = sp_out_flat  [gp*16 +: 16];
            assign exp_o [gp] = exp_out_flat [gp*16 +: 16];
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

    // RMSNorm registers — v2 (no pre-shift, raw x*x accumulator, finer ROM)
    // Each lane: |x|≤32767 → x² ≤ 2^30. Sum over CH_OUT*16 ≤ 128 lanes: ≤ 2^37. Use 40-bit.
    reg [39:0]        norm_sq_acc;
    reg signed [15:0] norm_S_reg;

    // CP-1: RMSNorm 2-step normalize unit (separate module, 1-cycle pipeline).
    // Inputs are the lane selected by mac_idx[3:0] from m_rd_data / c_rd_data,
    // S is the rsqrt result computed earlier (norm_S_reg). Output `rms_norm_out`
    // is valid 1 cycle after inputs present. Used by S_M1A_MAC and S_M1B_MAC.
    wire signed [15:0] rms_x_lane     = m_rd_data[mac_idx[3:0] * 16 +: 16];
    wire signed [15:0] rms_gamma_lane = c_rd_data[mac_idx[3:0] * 16 +: 16];
    wire signed [15:0] rms_norm_out;
    RMSNorm_Mul u_rmsnorm_mul (
        .clk        (clk),
        .x_in       (rms_x_lane),
        .gamma_in   (rms_gamma_lane),
        .S_in       (norm_S_reg),
        .x_norm_out (rms_norm_out)
    );
    // RMSNorm v2 mean-square → ROM index: total shift = log2_d + 2*FB - 1 - N
    // = log2_d + 15 (for FB=11, N=6).
    //   block 4   : CH_OUT=8, log2_d=7 → shift 22
    //   blocks 0-3: CH_OUT=4, log2_d=6 → shift 21
    wire [3:0]  log2_d_out    = (CH_OUT >= 4'd8) ? 4'd7 : 4'd6;
    wire [39:0] norm_mean_int = (CH_OUT >= 4'd8) ? (norm_sq_acc >> 22) : (norm_sq_acc >> 21);
    wire [12:0] norm_rom_idx  = (norm_mean_int > 40'd8191) ? 13'd8191 : norm_mean_int[12:0];

    // ----------------------------------------------------------------------
    // Const_Storage — every read-only / config storage in one hierarchy:
    //   • 48 activation LUT lanes (silu, softplus, exp; 16 each)
    //   • 8K×16 rsqrt ROM (RMSNorm)
    //   • 64×256 ram_const (per-block bias/scale/shift/gamma, DMA-loaded)
    // See Const_Storage.v.  ram_const takes the full DMA write port (it
    // consumes dma_target == 2'd3) and a registered core read port driven by
    // c_rd_addr — same 1-cycle latency the FSM previously expected.
    // ----------------------------------------------------------------------
    wire [15:0] rsqrt_rom_data;
    Const_Storage u_const (
        .clk             (clk),
        .silu_in_flat    (silu_in_flat),
        .sp_in_flat      (sp_in_flat),
        .exp_in_flat     (exp_in_flat),
        .silu_out_flat   (silu_out_flat),
        .sp_out_flat     (sp_out_flat),
        .exp_out_flat    (exp_out_flat),
        .rsqrt_idx       (norm_rom_idx),
        .rsqrt_data      (rsqrt_rom_data),
        .dma_write_en    (dma_write_en),
        .dma_target      (dma_target),
        .dma_addr        (dma_addr),
        .dma_wdata       (dma_wdata),
        .const_read_addr (c_rd_addr),
        .const_read_data (c_rd_data),
        .dma_read_en     (dma_read_en),
        .dma_rtarget     (dma_rtarget),
        .dma_raddr       (dma_raddr),
        .dma_rdata_const (const_dma_rdata)
    );

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
    // RMSNorm sub-phase states (before M1a and M1b, per-timestep)
    localparam S_NORM_M1A_SQ_READ  = 7'd115; localparam S_NORM_M1A_SQ_WAIT  = 7'd116;
    localparam S_NORM_M1A_SQ_LATCH = 7'd117; localparam S_NORM_M1A_SQ_NEXT  = 7'd118;
    localparam S_NORM_M1A_MEAN     = 7'd119;
    localparam S_NORM_M1B_SQ_READ  = 7'd120; localparam S_NORM_M1B_SQ_WAIT  = 7'd121;
    localparam S_NORM_M1B_SQ_LATCH = 7'd122; localparam S_NORM_M1B_SQ_NEXT  = 7'd123;
    localparam S_NORM_M1B_MEAN     = 7'd124;
    // Cascade write-back: copy (need_pool=0) or stride-2 MaxPool (need_pool=1).
    // Reads FINAL_OUT, writes A_INPUT_BASE for next block's P1.
    // BRAM_256b: dout_b is REGISTERED (1-cycle latency on read pipe), and the
    // address-mux is combinational from the m_rd_addr REG → effective 2-cycle
    // latency between m_rd_addr set and m_rd_data valid (matches S_BR_MAC pattern).
    // 5 states per (c_grp, t_out):
    //   RA: set addr_A          WA: wait        RB: latch m_rd_data=A, set addr_B (if pool)
    //   WB: wait                WR: m_rd_data=B (pool) or stale (copy); compute+write; advance
    localparam S_CASCADE_RA = 7'd125;
    localparam S_CASCADE_WA = 7'd87;
    localparam S_CASCADE_RB = 7'd126;
    localparam S_CASCADE_WB = 7'd88;
    localparam S_CASCADE_WR = 7'd127;

    // --------------------------------------------------------------
    // D2 end-to-end state numbers (8-bit range, > 127).
    //
    // ENCODER uses dedicated states (S_ENC_*) — the P1 stride/CH_IN signals
    // are wired to host-controlled values for the block, so swapping into
    // encoder-mode addressing in S_P1_MAC would race.  Dedicated states
    // keep P1 untouched and let the encoder use hardcoded d_in=12, d_out=64,
    // single-group input.
    // --------------------------------------------------------------
    localparam S_ENC_MAC       = 8'd128;
    localparam S_ENC_WAIT      = 8'd129;
    localparam S_ENC_WRITE     = 8'd130;
    localparam S_ENC_NEXT      = 8'd131;
    localparam S_ENC_DONE      = 8'd132;

    localparam S_GAP_READ      = 8'd133;
    localparam S_GAP_WAIT      = 8'd134;
    localparam S_GAP_LATCH     = 8'd135;
    localparam S_GAP_NEXT      = 8'd136;
    localparam S_GAP_FINALIZE  = 8'd137;
    localparam S_GAP_FIN_NEXT  = 8'd138;
    localparam S_GAP_DONE      = 8'd139;

    localparam S_FC_LOAD_BIAS  = 8'd140;
    localparam S_FC_BIAS_WAIT  = 8'd141;
    localparam S_FC_MAC        = 8'd142;
    localparam S_FC_WAIT       = 8'd143;
    localparam S_FC_NEXT_CLASS = 8'd144;
    localparam S_FC_FINALIZE   = 8'd145;
    localparam S_FC_DONE       = 8'd146;

    // ================================================================
    // FSM registers
    // ================================================================
    reg [7:0]  state;
    reg [9:0]  t_cnt;
    reg [3:0]  c_grp_m;
    reg [2:0]  c_grp;
    reg [7:0]  mac_idx;
    reg [5:0]  k_idx;
    reg [2:0]  substep;
    reg [255:0] max_buf;
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

    // --------------------------------------------------------------
    // D2 end-to-end control + accumulator registers
    // --------------------------------------------------------------
    reg        enc_phase;          // 1 = currently executing encoder (uses P1 datapath with swapped addrs)
    reg        enc_mode_reg;       // latched copy of top-level enc_mode at start
    reg        head_mode_reg;      // latched copy of top-level head_mode at start

    // GAP per-channel sum.  At T_GAP=250 max |sum| ≈ 250 × 32767 < 2^23,
    // so signed 24-bit suffices per lane.  Layout: gap_sum[c_grp][lane].
    reg signed [23:0] gap_sum [0:7][0:15];
    // GAP quantized output (16-bit per lane, packed 16 lanes per c_grp).
    reg        [255:0] gap_q_reg [0:7];
    reg [3:0]  gap_c_grp;         // current c_grp under accumulation (0..7)
    reg [9:0]  gap_t;             // current timestep (0..T_GAP-1)

    // FC accumulator: 5 classes × 1 lane each.  Accumulate Σ w[class, i] × gap[i]
    // for i = 0..127 in a 40-bit signed accumulator (matches Unified_PE width).
    reg signed [39:0] fc_acc;
    reg [2:0]  fc_class;          // current output class (0..4)
    reg [2:0]  fc_grp_in;         // current input group (0..7, 16 inputs per group)
    reg [3:0]  fc_lane;           // current input lane within group (0..15)
    reg signed [15:0] fc_bias_lane [0:4];   // latched FC bias per class
    // FC lane-serial timing fix: capture gap_q_reg[grp_in] once per grp_in so the
    // per-lane mux feeds the DSP off a flat 256-bit register, avoiding an 8-deep
    // 2D-array read in the same cycle as the multiplier.
    reg [255:0] fc_gap_word;

    // Cascade control regs (latched at start)
    reg        need_pool_reg, cascade_mode_reg;
    reg [9:0]  t_out_cnt;        // output timestep for cascade write (0..T_MAX/2-1 pool, 0..T_MAX-1 copy)
    wire [9:0] t_out_last_pool = (T_MAX >> 1) - 10'd1;     // T/2 - 1
    wire [9:0] t_out_last      = need_pool_reg ? t_out_last_pool : t_last;
    // Source-side t index into FINAL: pool reads even+odd of 2*t_out; copy reads t_out only.
    wire [9:0] src_t_a = need_pool_reg ? (t_out_cnt << 1) : t_out_cnt;
    wire [9:0] src_t_b = (t_out_cnt << 1) | 10'd1;          // only used if need_pool

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
    wire [9:0]  b1_t_prev   = (t_cnt == 10'd0)    ? 10'd0  : t_cnt - 10'd1;
    wire [9:0]  b1_t_next   = (t_cnt == t_last)   ? t_last : t_cnt + 10'd1;
    wire [255:0] b1_max_final = elem_max16(max_buf, m_rd_data);
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

    // CP-4 helper tasks — call alongside every `t_cnt <= ...` assignment so the
    // four address strides stay in lockstep without combinational multipliers.
    task t_cnt_zero;
        begin
            t_stride_in  <= 15'd0;
            t_stride_m   <= 15'd0;
            t_stride_out <= 15'd0;
            t_stride_xp  <= 15'd0;
        end
    endtask
    task t_cnt_inc;
        begin
            t_stride_in  <= t_stride_in  + {11'd0, CH_IN};
            t_stride_m   <= t_stride_m   + {7'd0,  ch_m_actual};
            t_stride_out <= t_stride_out + {11'd0, CH_OUT};
            t_stride_xp  <= t_stride_xp  + 15'd3;
        end
    endtask

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
            norm_sq_acc <= 40'd0; norm_S_reg <= 16'sd0;
            max_buf <= 256'd0;
            need_pool_reg <= 1'b0; cascade_mode_reg <= 1'b0;
            t_out_cnt <= 10'd0;
            t_stride_in <= 15'd0; t_stride_m <= 15'd0;
            t_stride_out <= 15'd0; t_stride_xp <= 15'd0;
            // D2 end-to-end resets
            enc_phase <= 1'b0; enc_mode_reg <= 1'b0; head_mode_reg <= 1'b0;
            done_encoder <= 1'b0; done_gap <= 1'b0; done_fc <= 1'b0;
            gap_c_grp <= 4'd0; gap_t <= 10'd0;
            fc_acc <= 40'sd0; fc_class <= 3'd0; fc_grp_in <= 3'd0; fc_lane <= 4'd0;
            fc_gap_word <= 256'd0;
            logit0 <= 16'sd0; logit1 <= 16'sd0; logit2 <= 16'sd0;
            logit3 <= 16'sd0; logit4 <= 16'sd0;
            for (i = 0; i < 16; i = i + 1) begin
                gap_sum[0][i] <= 24'sd0; gap_sum[1][i] <= 24'sd0;
                gap_sum[2][i] <= 24'sd0; gap_sum[3][i] <= 24'sd0;
                gap_sum[4][i] <= 24'sd0; gap_sum[5][i] <= 24'sd0;
                gap_sum[6][i] <= 24'sd0; gap_sum[7][i] <= 24'sd0;
            end
            for (i = 0; i < 8; i = i + 1) gap_q_reg[i] <= 256'd0;
            for (i = 0; i < 5; i = i + 1) fc_bias_lane[i] <= 16'sd0;
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
                    done_encoder <= 0; done_gap <= 0; done_fc <= 0;
                    t_cnt <= 0; c_grp <= 0; c_grp_m <= 0; mac_idx <= 0; t_cnt_zero;
                    k_idx <= 0; substep <= 0; branch_id <= 0; s_idx <= 0; c_grp_br <= 0;
                    // Latch end-to-end control for this run
                    enc_mode_reg     <= enc_mode;
                    head_mode_reg    <= head_mode;
                    need_pool_reg    <= need_pool;
                    cascade_mode_reg <= cascade_mode;
                    t_out_cnt        <= 10'd0;
                    // Reset GAP accumulators if head will run
                    for (i = 0; i < 16; i = i + 1) begin
                        gap_sum[0][i] <= 24'sd0; gap_sum[1][i] <= 24'sd0;
                        gap_sum[2][i] <= 24'sd0; gap_sum[3][i] <= 24'sd0;
                        gap_sum[4][i] <= 24'sd0; gap_sum[5][i] <= 24'sd0;
                        gap_sum[6][i] <= 24'sd0; gap_sum[7][i] <= 24'sd0;
                    end
                    gap_c_grp <= 4'd0;
                    gap_t     <= 10'd0;
                    fc_class  <= 3'd0;
                    fc_grp_in <= 3'd0;
                    fc_lane   <= 4'd0;
                    fc_acc    <= 40'sd0;
                    // Branch: encoder first if enc_mode, else P1 directly
                    if (enc_mode) begin
                        enc_phase <= 1'b1;
                        bank_sel  <= 1'b1;             // read ram_b (waveform), write ram_a (encoder out)
                        c_rd_addr <= C_ENC_BIAS;       // encoder bias base
                        state     <= S_ENC_MAC;
                    end else begin
                        enc_phase <= 1'b0;
                        bank_sel  <= 1'b0;
                        c_rd_addr <= C_P1_BIAS;
                        state     <= S_P1_MAC;
                    end
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
                        done_phase1 <= 1; t_cnt <= 0; t_cnt_zero; k_idx <= 0;
                        branch_id <= 0; bank_sel <= 1; c_grp_br <= 0;
                        state <= S_BR_MAC;
                    end else begin
                        t_cnt     <= t_cnt + 10'd1; t_cnt_inc;
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
                    3'd0: begin
                        if (branch_id == 3'd1) begin
                            // B1 MaxPool pass 1: read P1[t_prev]
                            m_rd_addr <= current_data_base
                                       + ({5'd0, b1_t_prev} * {11'd0, CH_OUT})
                                       + {10'd0, mac_idx[7:4]};
                        end else if (is_padding) begin
                            m_rd_addr <= 15'd0;
                        end else if (is_ch64_branch) begin
                            // Bot: d_out channels, CH_OUT words per t
                            m_rd_addr <= current_data_base
                                       + ({5'd0, t_eff} * {11'd0, CH_OUT})
                                       + {10'd0, mac_idx[7:4]};
                        end else begin
                            // B2/B3/B4: dim channels
                            m_rd_addr <= current_data_base
                                       + ({5'd0, t_eff} * br_dim_groups)
                                       + {14'd0, mac_idx[4]};
                        end
                        w_rd_addr <= current_w_base + br_w_offset
                                   + ({9'd0, k_idx} * {7'd0, current_num_in_ch})
                                   + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 3'd1;
                    end
                    3'd1: begin pe_A <= 16'sd0; substep <= 3'd2; end
                    3'd2: begin
                        if (branch_id == 3'd1) begin
                            // B1 MaxPool: save P1[t_prev]; issue P1[t_cnt]
                            max_buf   <= m_rd_data;
                            m_rd_addr <= current_data_base
                                       + ({5'd0, t_cnt}    * {11'd0, CH_OUT})
                                       + {10'd0, mac_idx[7:4]};
                            substep <= 3'd3;
                        end else begin
                            if (is_padding) pe_A <= 16'sd0;
                            else            pe_A <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                            pe_B     <= w_rd_data;
                            pe_clear <= (k_idx == 6'd0 && mac_idx == 8'd0);
                            if (mac_idx == mac_target[7:0]) begin
                                mac_idx <= 0;
                                if (k_idx == k_target) state <= S_BR_WAIT;
                                else begin k_idx <= k_idx + 6'd1; substep <= 3'd0; end
                            end else begin
                                mac_idx <= mac_idx + 8'd1; substep <= 3'd0;
                            end
                        end
                    end
                    3'd3: begin substep <= 3'd4; end  // B1: wait for P1[t_cnt]
                    3'd4: begin
                        // B1: max(t_prev, t_curr); issue P1[t_next]
                        max_buf   <= elem_max16(max_buf, m_rd_data);
                        m_rd_addr <= current_data_base
                                   + ({5'd0, b1_t_next} * {11'd0, CH_OUT})
                                   + {10'd0, mac_idx[7:4]};
                        substep <= 3'd5;
                    end
                    3'd5: begin substep <= 3'd6; end  // B1: wait for P1[t_next]
                    3'd6: begin
                        // B1: final max, MAC
                        pe_A     <= b1_max_final[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == mac_target[7:0]) begin
                            mac_idx <= 0;
                            if (k_idx == k_target) state <= S_BR_WAIT;
                            else begin k_idx <= k_idx + 6'd1; substep <= 3'd0; end
                        end else begin
                            mac_idx <= mac_idx + 8'd1; substep <= 3'd0;
                        end
                    end
                    default: substep <= 3'd0;
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
                    t_cnt <= 0; t_cnt_zero;
                    if (c_grp_br != br_grp_last) begin
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
                                t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; mac_idx <= 0;
                                c_grp <= 0; norm_sq_acc <= 40'd0;
                                bank_sel <= 1;
                                state    <= S_NORM_M1A_SQ_READ;
                            end
                            default: state <= S_DONE;
                        endcase
                    end
                end else begin
                    t_cnt <= t_cnt + 10'd1; t_cnt_inc;
                    state <= S_BR_MAC;
                end
            end

            // --------------------------------------------------------
            // M1a : in_proj_x   d_out -> d_inner
            // --------------------------------------------------------
            // CP-1: 4-substep inner loop. substep 2 presents inputs to
            // RMSNorm_Mul (its internal p1_reg latches at end of cycle);
            // substep 3 captures the now-valid `rms_norm_out` into pe_A
            // together with pe_B/pe_clear (all aligned for PE_Array MAC).
            S_M1A_MAC: begin
                case (substep)
                    3'd0: begin
                        m_rd_addr <= B_P1_OUT + t_stride_out + {10'd0, mac_idx[7:4]};
                        w_rd_addr <= W_M_X_BASE + ({7'd0, c_grp_m} * {7'd0, d_out}) + {7'd0, mac_idx};
                        c_rd_addr <= C_NORM_W + {12'd0, mac_idx[7:4]};
                        pe_A <= 16'sd0; substep <= 3'd1;
                    end
                    3'd1: begin pe_A <= 16'sd0; substep <= 3'd2; end
                    3'd2: begin
                        // Inputs (rms_x_lane, rms_gamma_lane, norm_S_reg) feed
                        // RMSNorm_Mul combinationally; its p1_reg latches now.
                        substep <= 3'd3;
                    end
                    3'd3: begin
                        pe_A     <= rms_norm_out;
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == d_out - 8'd1) state <= S_M1A_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 3'd0; end
                    end
                    default: substep <= 3'd0;
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; c_grp <= 0; norm_sq_acc <= 40'd0;
                        state <= S_NORM_M1B_SQ_READ;
                    end else begin
                        t_cnt <= t_cnt + 10'd1; t_cnt_inc; c_grp <= 0; norm_sq_acc <= 40'd0;
                        state <= S_NORM_M1A_SQ_READ;
                    end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M1A_MAC; end
            end

            // --------------------------------------------------------
            // RMSNorm sub-phase for M1a: sum-of-squares over B_P1_OUT[t]
            // --------------------------------------------------------
            S_NORM_M1A_SQ_READ: begin
                m_rd_addr <= B_P1_OUT + t_stride_out + {12'd0, c_grp};
                state     <= S_NORM_M1A_SQ_WAIT;
            end
            S_NORM_M1A_SQ_WAIT:  state <= S_NORM_M1A_SQ_LATCH;
            S_NORM_M1A_SQ_LATCH: begin
                norm_sq_acc <= norm_sq_acc + norm_sq16_fn(m_rd_data);
                state       <= S_NORM_M1A_SQ_NEXT;
            end
            S_NORM_M1A_SQ_NEXT: begin
                if (c_grp == ch_out_last[2:0]) begin
                    c_grp <= 0; state <= S_NORM_M1A_MEAN;
                end else begin
                    c_grp <= c_grp + 3'd1; state <= S_NORM_M1A_SQ_READ;
                end
            end
            S_NORM_M1A_MEAN: begin
                norm_S_reg <= $signed(rsqrt_rom_data);
                c_grp <= 0; state <= S_M1A_MAC;
            end

            // --------------------------------------------------------
            // M1b : in_proj_z   d_out -> d_inner  -> A_Z_GATE
            // --------------------------------------------------------
            // CP-1: same 4-substep structure as M1A_MAC (see explanation above).
            S_M1B_MAC: begin
                case (substep)
                    3'd0: begin
                        m_rd_addr <= B_P1_OUT + t_stride_out + {10'd0, mac_idx[7:4]};
                        w_rd_addr <= W_M_Z_BASE + ({7'd0, c_grp_m} * {7'd0, d_out}) + {7'd0, mac_idx};
                        c_rd_addr <= C_NORM_W + {12'd0, mac_idx[7:4]};
                        pe_A <= 16'sd0; substep <= 3'd1;
                    end
                    3'd1: begin pe_A <= 16'sd0; substep <= 3'd2; end
                    3'd2: begin
                        // RMSNorm_Mul Stage 1 register latches at end of cycle.
                        substep <= 3'd3;
                    end
                    3'd3: begin
                        pe_A     <= rms_norm_out;
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == d_out - 8'd1) state <= S_M1B_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 3'd0; end
                    end
                    default: substep <= 3'd0;
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; k_idx <= 0; bank_sel <= 0; state <= S_M2_MAC;
                    end else begin
                        t_cnt <= t_cnt + 10'd1; t_cnt_inc; c_grp <= 0; norm_sq_acc <= 40'd0;
                        state <= S_NORM_M1B_SQ_READ;
                    end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M1B_MAC; end
            end

            // --------------------------------------------------------
            // RMSNorm sub-phase for M1b: sum-of-squares over B_P1_OUT[t]
            // --------------------------------------------------------
            S_NORM_M1B_SQ_READ: begin
                m_rd_addr <= B_P1_OUT + t_stride_out + {12'd0, c_grp};
                state     <= S_NORM_M1B_SQ_WAIT;
            end
            S_NORM_M1B_SQ_WAIT:  state <= S_NORM_M1B_SQ_LATCH;
            S_NORM_M1B_SQ_LATCH: begin
                norm_sq_acc <= norm_sq_acc + norm_sq16_fn(m_rd_data);
                state       <= S_NORM_M1B_SQ_NEXT;
            end
            S_NORM_M1B_SQ_NEXT: begin
                if (c_grp == ch_out_last[2:0]) begin
                    c_grp <= 0; state <= S_NORM_M1B_MEAN;
                end else begin
                    c_grp <= c_grp + 3'd1; state <= S_NORM_M1B_SQ_READ;
                end
            end
            S_NORM_M1B_MEAN: begin
                norm_S_reg <= $signed(rsqrt_rom_data);
                c_grp <= 0; state <= S_M1B_MAC;
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; bank_sel <= 1; state <= S_M3_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_M2_MAC; end
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; substep <= 0; bank_sel <= 0; state <= S_M3CP_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_M3_READ; end
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
                        t_cnt <= 0; t_cnt_zero; c_grp <= 0; mac_idx <= 0; substep <= 0; bank_sel <= 0; state <= S_M4_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_M3CP_READ; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M3CP_READ; end
            end

            // --------------------------------------------------------
            // M4 : x_proj  d_inner -> 48  (3 groups fixed)
            // --------------------------------------------------------
            S_M4_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= A_X_INNER + t_stride_m + {7'd0, mac_idx[7:4]};
                        w_rd_addr <= W_XPROJ_BASE_W + ({12'd0, c_grp} * {6'd0, d_inner}) + {7'd0, mac_idx};
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; mac_idx <= 0; k_idx <= 0; substep <= 0; bank_sel <= 1; state <= S_M5_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_M4_MAC; end
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; s_idx <= 0; bank_sel <= 1; state <= S_M6A_INIT_H;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_M5_MAC; end
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; s_idx <= 0; bank_sel <= 1; state <= S_M6A_DA_READ;
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
                        t_cnt <= 0; t_cnt_zero; c_grp_m <= 0; bank_sel <= 1; state <= S_M7_RY_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; s_idx <= 0; state <= S_M6A_DA_READ; end
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
                        t_cnt <= 0; t_cnt_zero; c_grp <= 0; mac_idx <= 0; substep <= 0; bank_sel <= 1; state <= S_M8_MAC;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_M7_RY_READ; end
                end else begin c_grp_m <= c_grp_m + 4'd1; state <= S_M7_RY_READ; end
            end

            // --------------------------------------------------------
            // M8 : out_proj  d_inner -> d_out
            // --------------------------------------------------------
            S_M8_MAC: begin
                case (substep)
                    2'd0: begin
                        m_rd_addr <= B_Y_SSM + t_stride_m + {7'd0, mac_idx[7:4]};
                        w_rd_addr <= W_OUTPROJ_BASE + ({11'd0, c_grp} * {6'd0, d_inner}) + {7'd0, mac_idx};
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
                        t_cnt <= 0; t_cnt_zero; c_grp <= 0; substep <= 0;
                        bank_sel  <= 0; m_we <= 0;
                        c_rd_addr <= C_INC_SCALE;
                        state     <= S_FIN_READ;
                    end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; bank_sel <= 1; state <= S_M8_MAC; end
                end else begin c_grp <= c_grp + 3'd1; bank_sel <= 1; state <= S_M8_MAC; end
            end

            // --------------------------------------------------------
            // PHASE 4 : Final  relu(bn(inception)) + relu(mamba_out)
            //          (PyTorch ITMBlock formula — see ITMN.py:123-128)
            //   x1 = bn_relu(incep, scale, shift)   �? inception's BN+ReLU
            //   x2 = relu16(mamba_out)              �? mamba branch ReLU
            //   out = sat_add16(x1, x2)
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
                m_wr_data[  0 +: 16] <= sat_add16(bn_relu(incep_reg[  0 +: 16], m_wr_data[  0 +: 16], c_rd_data[  0 +: 16]), relu16(m_rd_data[  0 +: 16]));
                m_wr_data[ 16 +: 16] <= sat_add16(bn_relu(incep_reg[ 16 +: 16], m_wr_data[ 16 +: 16], c_rd_data[ 16 +: 16]), relu16(m_rd_data[ 16 +: 16]));
                m_wr_data[ 32 +: 16] <= sat_add16(bn_relu(incep_reg[ 32 +: 16], m_wr_data[ 32 +: 16], c_rd_data[ 32 +: 16]), relu16(m_rd_data[ 32 +: 16]));
                m_wr_data[ 48 +: 16] <= sat_add16(bn_relu(incep_reg[ 48 +: 16], m_wr_data[ 48 +: 16], c_rd_data[ 48 +: 16]), relu16(m_rd_data[ 48 +: 16]));
                m_wr_data[ 64 +: 16] <= sat_add16(bn_relu(incep_reg[ 64 +: 16], m_wr_data[ 64 +: 16], c_rd_data[ 64 +: 16]), relu16(m_rd_data[ 64 +: 16]));
                m_wr_data[ 80 +: 16] <= sat_add16(bn_relu(incep_reg[ 80 +: 16], m_wr_data[ 80 +: 16], c_rd_data[ 80 +: 16]), relu16(m_rd_data[ 80 +: 16]));
                m_wr_data[ 96 +: 16] <= sat_add16(bn_relu(incep_reg[ 96 +: 16], m_wr_data[ 96 +: 16], c_rd_data[ 96 +: 16]), relu16(m_rd_data[ 96 +: 16]));
                m_wr_data[112 +: 16] <= sat_add16(bn_relu(incep_reg[112 +: 16], m_wr_data[112 +: 16], c_rd_data[112 +: 16]), relu16(m_rd_data[112 +: 16]));
                m_wr_data[128 +: 16] <= sat_add16(bn_relu(incep_reg[128 +: 16], m_wr_data[128 +: 16], c_rd_data[128 +: 16]), relu16(m_rd_data[128 +: 16]));
                m_wr_data[144 +: 16] <= sat_add16(bn_relu(incep_reg[144 +: 16], m_wr_data[144 +: 16], c_rd_data[144 +: 16]), relu16(m_rd_data[144 +: 16]));
                m_wr_data[160 +: 16] <= sat_add16(bn_relu(incep_reg[160 +: 16], m_wr_data[160 +: 16], c_rd_data[160 +: 16]), relu16(m_rd_data[160 +: 16]));
                m_wr_data[176 +: 16] <= sat_add16(bn_relu(incep_reg[176 +: 16], m_wr_data[176 +: 16], c_rd_data[176 +: 16]), relu16(m_rd_data[176 +: 16]));
                m_wr_data[192 +: 16] <= sat_add16(bn_relu(incep_reg[192 +: 16], m_wr_data[192 +: 16], c_rd_data[192 +: 16]), relu16(m_rd_data[192 +: 16]));
                m_wr_data[208 +: 16] <= sat_add16(bn_relu(incep_reg[208 +: 16], m_wr_data[208 +: 16], c_rd_data[208 +: 16]), relu16(m_rd_data[208 +: 16]));
                m_wr_data[224 +: 16] <= sat_add16(bn_relu(incep_reg[224 +: 16], m_wr_data[224 +: 16], c_rd_data[224 +: 16]), relu16(m_rd_data[224 +: 16]));
                m_wr_data[240 +: 16] <= sat_add16(bn_relu(incep_reg[240 +: 16], m_wr_data[240 +: 16], c_rd_data[240 +: 16]), relu16(m_rd_data[240 +: 16]));
                state <= S_FIN_NEXT;
            end
            S_FIN_NEXT: begin
                m_we <= 0;
                if (t_cnt == t_last) begin
                    t_cnt <= 0; t_cnt_zero;
                    if (c_grp == ch_out_last[2:0]) begin
                        // FIN complete. Branch on cascade / head mode:
                        //   cascade_mode=1                : write back to A_INPUT_BASE
                        //   cascade_mode=0 + head_mode=1  : run GAP + FC (D2 end-to-end)
                        //   cascade_mode=0 + head_mode=0  : terminal, host reads FINAL_OUT
                        if (cascade_mode_reg) begin
                            c_grp     <= 3'd0;
                            t_out_cnt <= 10'd0;
                            bank_sel  <= 1'b0;          // FINAL c_grp=0 lives in bank 0
                            state     <= S_CASCADE_RA;
                        end else if (head_mode_reg) begin
                            // D2 GAP entry: start from c_grp=0, t=0; reset accumulators
                            gap_c_grp <= 4'd0;
                            gap_t     <= 10'd0;
                            for (i = 0; i < 16; i = i + 1) begin
                                gap_sum[0][i] <= 24'sd0; gap_sum[1][i] <= 24'sd0;
                                gap_sum[2][i] <= 24'sd0; gap_sum[3][i] <= 24'sd0;
                                gap_sum[4][i] <= 24'sd0; gap_sum[5][i] <= 24'sd0;
                                gap_sum[6][i] <= 24'sd0; gap_sum[7][i] <= 24'sd0;
                            end
                            state     <= S_GAP_READ;
                        end else begin
                            done_all <= 1; state <= S_DONE;
                        end
                    end else begin
                        c_grp    <= c_grp + 3'd1;
                        bank_sel <= 1;
                        state    <= S_FIN_READ;
                    end
                end else begin t_cnt <= t_cnt + 10'd1; t_cnt_inc; state <= S_FIN_READ; end
            end

            // --------------------------------------------------------
            // Cascade write-back  (copy or stride-2 MaxPool)
            //   Outer loop: c_grp = 0..ch_out_last
            //   Inner loop: t_out_cnt = 0..t_out_last
            //     READ FINAL[c_grp][src_t_a] — c_grp=0 lives in ram_b (B_FINAL_OUT),
            //                                  c_grp>=1 lives in ram_a (A_FINAL_OUT)
            //     if need_pool: also read FINAL[c_grp][src_t_b] and take elem_max16
            //     WRITE result to A_INPUT_BASE in ram_a + t_out_cnt*CH_OUT + c_grp
            //
            // Memory_System bank_sel routing (asymmetric):
            //   read : bank_sel=0 → ram_a;  bank_sel=1 → ram_b
            //   write: bank_sel=0 → ram_b;  bank_sel=1 → ram_a   (we_a triggered by bank_sel==1)
            // --------------------------------------------------------
            S_CASCADE_RA: begin
                // bank_sel for READ: c_grp=0 → ram_b (set 1); c_grp>=1 → ram_a (set 0)
                bank_sel  <= (c_grp == 3'd0) ? 1'b1 : 1'b0;
                m_rd_addr <= ((c_grp == 3'd0) ? B_FINAL_OUT : A_FINAL_OUT)
                           + ({5'd0, src_t_a} * {11'd0, CH_OUT})
                           + {12'd0, c_grp};
                state     <= S_CASCADE_WA;
            end

            S_CASCADE_WA: state <= S_CASCADE_RB;   // wait 1 cycle for BRAM dout_b

            S_CASCADE_RB: begin
                // m_rd_data here = FINAL[c_grp][src_t_a].  bank_sel held from RA.
                max_buf <= m_rd_data;
                if (need_pool_reg) begin
                    m_rd_addr <= ((c_grp == 3'd0) ? B_FINAL_OUT : A_FINAL_OUT)
                               + ({5'd0, src_t_b} * {11'd0, CH_OUT})
                               + {12'd0, c_grp};
                end
                state <= S_CASCADE_WB;
            end

            S_CASCADE_WB: state <= S_CASCADE_WR;   // wait 1 cycle for BRAM dout_b (B read, pool only)

            S_CASCADE_WR: begin
                // m_rd_data this cycle = FINAL[c_grp][src_t_b]  (pool) or unused (copy).
                // Drive write: bank_sel=1 routes core_write to ram_a (A_INPUT_BASE).
                m_we      <= 1'b1;
                bank_sel  <= 1'b1;
                m_wr_addr <= A_INPUT_BASE
                           + ({5'd0, t_out_cnt} * {11'd0, CH_OUT})
                           + {12'd0, c_grp};
                m_wr_data <= need_pool_reg ? elem_max16(max_buf, m_rd_data) : max_buf;

                // Advance counters inline
                if (t_out_cnt == t_out_last) begin
                    t_out_cnt <= 10'd0;
                    if (c_grp == ch_out_last[2:0]) begin
                        done_all <= 1'b1;
                        state    <= S_DONE;
                    end else begin
                        c_grp <= c_grp + 3'd1;
                        state <= S_CASCADE_RA;
                    end
                end else begin
                    t_out_cnt <= t_out_cnt + 10'd1;
                    state     <= S_CASCADE_RA;
                end
            end

            // ========================================================
            // D2 — ENCODER PHASE  (Conv1d(12, 64, k=1) + BN, fused)
            //
            // Reuses PE_Array MAC + Const_Storage bias + sat_add16.  Address
            // layout differs from P1: input from B_ENC_IN_BASE (ram_b), weight
            // from W_ENC_BASE, bias from C_ENC_BIAS, output to A_INPUT_BASE
            // (ram_a) so block 0's P1 reads it naturally on the next start.
            //
            // Loop nest (same shape as P1 but CH_IN=1 hardcoded):
            //   t = 0..T_ENC-1
            //     c_grp = 0..3        (d_out=64, 16 lanes/group)
            //       mac_idx = 0..11   (12 input channels; lanes 12..15 zero-padded)
            //         3-substep PE_Array MAC (matches S_P1_MAC pattern)
            //
            // bank_sel = 1 throughout (read ram_b, write ram_a).
            // ========================================================
            S_ENC_MAC: begin
                case (substep)
                    2'd0: begin
                        // Encoder input: 1 word per timestep at B_ENC_IN_BASE+t
                        m_rd_addr <= B_ENC_IN_BASE + {5'd0, t_cnt};
                        // Encoder weight: one row (16-lane word) per output channel
                        // Row index = c_grp*16 + mac_idx ; lanes 0..11 valid, 12..15 = 0
                        w_rd_addr <= W_ENC_BASE
                                   + ({7'd0, c_grp} * 15'd16)
                                   + {7'd0, mac_idx};
                        pe_A <= 16'sd0; substep <= 2'd1;
                    end
                    2'd1: begin pe_A <= 16'sd0; substep <= 2'd2; end
                    2'd2: begin
                        // mac_idx ∈ [0, 11] indexes into m_rd_data lanes 0..11.
                        // For mac_idx ≥ 12 we would broadcast 0 (padding) — but
                        // our loop only iterates 0..11 so this never happens.
                        pe_A     <= m_rd_data[mac_idx[3:0] * 16 +: 16];
                        pe_B     <= w_rd_data;
                        pe_clear <= (mac_idx == 8'd0);
                        if (mac_idx == 8'd11) state <= S_ENC_WAIT;
                        else begin mac_idx <= mac_idx + 8'd1; substep <= 2'd0; end
                    end
                    default: substep <= 2'd0;
                endcase
            end

            S_ENC_WAIT:  state <= S_ENC_WRITE;

            S_ENC_WRITE: begin
                // Write encoder output to A_INPUT_BASE for block 0's P1 to read.
                // Stride = 4 words/timestep (d_out=64 channels / 16 lanes = 4 c_grp).
                m_we      <= 1'b1;
                bank_sel  <= 1'b1;     // write ram_a
                m_wr_addr <= A_INPUT_BASE
                           + ({5'd0, t_cnt} * 15'd4)     // 4 c_grp words per t
                           + {12'd0, c_grp};
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
                state <= S_ENC_NEXT;
            end

            S_ENC_NEXT: begin
                mac_idx <= 8'd0; substep <= 0;
                if (c_grp == 3'd3) begin
                    // Done all 4 c_grp for this t — advance t.
                    c_grp <= 3'd0;
                    if (t_cnt == T_ENC - 10'd1) begin
                        // Encoder done — transition to block 0's P1 normally.
                        done_encoder <= 1'b1;
                        enc_phase    <= 1'b0;
                        t_cnt        <= 10'd0; t_cnt_zero;
                        bank_sel     <= 1'b0;          // back to P1 read ram_a / write ram_b
                        c_rd_addr    <= C_P1_BIAS;
                        // P1 control reset (consistent with S_IDLE→P1 transition)
                        need_pool_reg    <= need_pool;
                        cascade_mode_reg <= cascade_mode;
                        t_out_cnt        <= 10'd0;
                        state            <= S_P1_MAC;
                    end else begin
                        t_cnt     <= t_cnt + 10'd1;
                        // Encoder uses 1-word/t stride — t_stride_* not used here.
                        c_rd_addr <= C_ENC_BIAS;
                        state     <= S_ENC_MAC;
                    end
                end else begin
                    c_grp     <= c_grp + 3'd1;
                    c_rd_addr <= C_ENC_BIAS + {12'd0, c_grp} + 15'd1;
                    state     <= S_ENC_MAC;
                end
            end

            // ========================================================
            // D2 — GAP PHASE  (per-channel mean over T_GAP timesteps)
            //
            // Iterate c_grp = 0..7, t = 0..T_GAP-1.  For each (c_grp, t) read
            // 1 word (16 lanes of 16-bit) from FINAL_OUT and add each lane's
            // signed value to gap_sum[c_grp][lane] (24-bit accumulator).
            //
            // FINAL_OUT split (matches FIN_WRITE addressing):
            //   c_grp=0     → B_FINAL_OUT (ram_b)
            //   c_grp>=1    → A_FINAL_OUT (ram_a)
            // Address: base + t * CH_OUT + c_grp
            //
            // After T_GAP timesteps for all 8 c_grp: finalize via INV_T_Q15.
            // ========================================================
            S_GAP_READ: begin
                bank_sel  <= (gap_c_grp == 4'd0) ? 1'b1 : 1'b0;   // read bank
                m_rd_addr <= ((gap_c_grp == 4'd0) ? B_FINAL_OUT : A_FINAL_OUT)
                           + ({5'd0, gap_t} * {11'd0, CH_OUT})
                           + {11'd0, gap_c_grp};
                state     <= S_GAP_WAIT;
            end

            S_GAP_WAIT: state <= S_GAP_LATCH;   // 1-cycle BRAM dout_b latency

            S_GAP_LATCH: begin
                // Accumulate each lane (signed 16-bit → sign-extended to 24-bit).
                gap_sum[gap_c_grp[2:0]][ 0] <= gap_sum[gap_c_grp[2:0]][ 0]
                    + {{8{m_rd_data[ 15]}}, m_rd_data[  0 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 1] <= gap_sum[gap_c_grp[2:0]][ 1]
                    + {{8{m_rd_data[ 31]}}, m_rd_data[ 16 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 2] <= gap_sum[gap_c_grp[2:0]][ 2]
                    + {{8{m_rd_data[ 47]}}, m_rd_data[ 32 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 3] <= gap_sum[gap_c_grp[2:0]][ 3]
                    + {{8{m_rd_data[ 63]}}, m_rd_data[ 48 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 4] <= gap_sum[gap_c_grp[2:0]][ 4]
                    + {{8{m_rd_data[ 79]}}, m_rd_data[ 64 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 5] <= gap_sum[gap_c_grp[2:0]][ 5]
                    + {{8{m_rd_data[ 95]}}, m_rd_data[ 80 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 6] <= gap_sum[gap_c_grp[2:0]][ 6]
                    + {{8{m_rd_data[111]}}, m_rd_data[ 96 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 7] <= gap_sum[gap_c_grp[2:0]][ 7]
                    + {{8{m_rd_data[127]}}, m_rd_data[112 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 8] <= gap_sum[gap_c_grp[2:0]][ 8]
                    + {{8{m_rd_data[143]}}, m_rd_data[128 +: 16]};
                gap_sum[gap_c_grp[2:0]][ 9] <= gap_sum[gap_c_grp[2:0]][ 9]
                    + {{8{m_rd_data[159]}}, m_rd_data[144 +: 16]};
                gap_sum[gap_c_grp[2:0]][10] <= gap_sum[gap_c_grp[2:0]][10]
                    + {{8{m_rd_data[175]}}, m_rd_data[160 +: 16]};
                gap_sum[gap_c_grp[2:0]][11] <= gap_sum[gap_c_grp[2:0]][11]
                    + {{8{m_rd_data[191]}}, m_rd_data[176 +: 16]};
                gap_sum[gap_c_grp[2:0]][12] <= gap_sum[gap_c_grp[2:0]][12]
                    + {{8{m_rd_data[207]}}, m_rd_data[192 +: 16]};
                gap_sum[gap_c_grp[2:0]][13] <= gap_sum[gap_c_grp[2:0]][13]
                    + {{8{m_rd_data[223]}}, m_rd_data[208 +: 16]};
                gap_sum[gap_c_grp[2:0]][14] <= gap_sum[gap_c_grp[2:0]][14]
                    + {{8{m_rd_data[239]}}, m_rd_data[224 +: 16]};
                gap_sum[gap_c_grp[2:0]][15] <= gap_sum[gap_c_grp[2:0]][15]
                    + {{8{m_rd_data[255]}}, m_rd_data[240 +: 16]};
                state <= S_GAP_NEXT;
            end

            S_GAP_NEXT: begin
                if (gap_t == T_GAP - 10'd1) begin
                    gap_t <= 10'd0;
                    if (gap_c_grp == {1'b0, ch_out_last[2:0]}) begin
                        gap_c_grp <= 4'd0;
                        state     <= S_GAP_FINALIZE;
                    end else begin
                        gap_c_grp <= gap_c_grp + 4'd1;
                        state     <= S_GAP_READ;
                    end
                end else begin
                    gap_t <= gap_t + 10'd1;
                    state <= S_GAP_READ;
                end
            end

            // Multiply each gap_sum[c_grp][lane] by INV_T_Q15, >> 15, sat16.
            // Process 1 c_grp per cycle (gap_c_grp iterates 0..7).
            S_GAP_FINALIZE: begin
                gap_q_reg[gap_c_grp[2:0]][  0 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 0]);
                gap_q_reg[gap_c_grp[2:0]][ 16 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 1]);
                gap_q_reg[gap_c_grp[2:0]][ 32 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 2]);
                gap_q_reg[gap_c_grp[2:0]][ 48 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 3]);
                gap_q_reg[gap_c_grp[2:0]][ 64 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 4]);
                gap_q_reg[gap_c_grp[2:0]][ 80 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 5]);
                gap_q_reg[gap_c_grp[2:0]][ 96 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 6]);
                gap_q_reg[gap_c_grp[2:0]][112 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 7]);
                gap_q_reg[gap_c_grp[2:0]][128 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 8]);
                gap_q_reg[gap_c_grp[2:0]][144 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][ 9]);
                gap_q_reg[gap_c_grp[2:0]][160 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][10]);
                gap_q_reg[gap_c_grp[2:0]][176 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][11]);
                gap_q_reg[gap_c_grp[2:0]][192 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][12]);
                gap_q_reg[gap_c_grp[2:0]][208 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][13]);
                gap_q_reg[gap_c_grp[2:0]][224 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][14]);
                gap_q_reg[gap_c_grp[2:0]][240 +: 16] <= sat_mul_q15(gap_sum[gap_c_grp[2:0]][15]);
                state <= S_GAP_FIN_NEXT;
            end

            S_GAP_FIN_NEXT: begin
                if (gap_c_grp == {1'b0, ch_out_last[2:0]}) begin
                    gap_c_grp <= 4'd0;
                    done_gap  <= 1'b1;
                    // Transition to FC: load class-0 bias.
                    fc_class  <= 3'd0;
                    fc_grp_in <= 3'd0;
                    fc_lane   <= 4'd0;
                    fc_acc    <= 40'sd0;
                    c_rd_addr <= C_FC_BIAS;
                    state     <= S_FC_LOAD_BIAS;
                end else begin
                    gap_c_grp <= gap_c_grp + 4'd1;
                    state     <= S_GAP_FINALIZE;
                end
            end

            S_GAP_DONE: begin
                // Reserved for direct GAP-only operation (head_mode=1 without FC).
                // Currently always falls through to FC; placeholder for future use.
                state <= S_IDLE;
            end

            // ========================================================
            // D2 — FC PHASE  (Linear(128, 5) with sat_add16 bias)
            //
            // For each class c ∈ 0..4:
            //   fc_acc = sum over grp_in=0..7, lane=0..15 of (W_fc[c, grp_in*16+lane] * gap_q[grp_in, lane])
            //   logit[c] = sat_add16(sat16(fc_acc >> FB), fc_bias[c])
            //
            // W_FC layout in ram_weight: W_FC_BASE + class*8 + grp_in
            //   1 word per (class, grp_in) holding 16 lanes.
            //
            // FC bias layout in ram_const: C_FC_BIAS, lanes [0..4] valid, [5..15] = 0.
            //
            // PE_Array is not used here (data already in gap_q_reg).  We use a single
            // 40-bit accumulator + 1 DSP-mappable multiplier per cycle.  128 cycles
            // per class × 5 classes = 640 cycles total.
            // ========================================================
            S_FC_LOAD_BIAS: begin
                state <= S_FC_BIAS_WAIT;
            end

            S_FC_BIAS_WAIT: begin
                // c_rd_data now holds FC bias word: lanes [0..4] = bias[0..4]
                fc_bias_lane[0] <= c_rd_data[  0 +: 16];
                fc_bias_lane[1] <= c_rd_data[ 16 +: 16];
                fc_bias_lane[2] <= c_rd_data[ 32 +: 16];
                fc_bias_lane[3] <= c_rd_data[ 48 +: 16];
                fc_bias_lane[4] <= c_rd_data[ 64 +: 16];
                // Issue first weight read for (class=0, grp_in=0)
                w_rd_addr <= W_FC_BASE + ({12'd0, fc_class} * 15'd8) + {12'd0, fc_grp_in};
                fc_acc    <= 40'sd0;
                substep   <= 2'd0;
                state     <= S_FC_MAC;
            end

            // Per-class MAC: scan grp_in = 0..7, lane = 0..15 (= 128 inputs).
            // 3-substep pattern to give w_rd_data 1-cycle latency:
            //   substep 0: issue w_rd_addr for (fc_class, fc_grp_in)
            //   substep 1: wait
            //   substep 2: w_rd_data valid, MAC across 16 lanes of this grp_in
            // Timing fix: the original code summed 16 lane products in one cycle,
            // which Vivado packed into a 16-DSP-ALU cascade (43 logic levels, WNS
            // -9.66 ns @ 10 ns).  Serialized version below: one lane per cycle.
            //
            //   substep 0  : issue w_rd_addr for (class, grp_in); latch gap word
            //   substep 1  : wait for BRAM (w_rd_data valid next cycle)
            //   substep 2  : 16-cycle inner loop (fc_lane = 0..15)
            //                MAC one lane per cycle into fc_acc via a single DSP.
            //
            // Cycle cost: 18/grp_in × 8 grp_in × 5 class ≈ 720 cycles for FC, up
            // from ~150 before.  Block 4 dominates at ~12 M cycles so overhead
            // <0.01% per sample.  Resource saved: 16 → 1 DSP for FC.
            S_FC_MAC: begin
                case (substep)
                    2'd0: begin
                        w_rd_addr     <= W_FC_BASE
                                       + ({12'd0, fc_class} * 15'd8)
                                       + {12'd0, fc_grp_in};
                        // Capture this grp_in's GAP word so the per-lane mux is
                        // off a flat 256-bit register, not an 8-deep 2D array
                        // read each cycle.
                        fc_gap_word   <= gap_q_reg[fc_grp_in];
                        fc_lane       <= 4'd0;
                        substep       <= 2'd1;
                    end
                    2'd1: substep <= 2'd2;          // wait BRAM read
                    2'd2: begin
                        // Single-lane MAC: 1 mul + 40-bit add per cycle.
                        fc_acc <= fc_acc
                                + ($signed(w_rd_data [fc_lane * 16 +: 16])
                                 * $signed(fc_gap_word[fc_lane * 16 +: 16]));
                        if (fc_lane == 4'd15) begin
                            fc_lane <= 4'd0;
                            if (fc_grp_in == 3'd7) state <= S_FC_WAIT;
                            else begin
                                fc_grp_in <= fc_grp_in + 3'd1;
                                substep   <= 2'd0;
                            end
                        end else begin
                            fc_lane <= fc_lane + 4'd1;
                            // Stay in substep 2 — w_rd_addr / fc_gap_word held.
                        end
                    end
                    default: substep <= 2'd0;
                endcase
            end

            S_FC_WAIT: state <= S_FC_NEXT_CLASS;

            S_FC_NEXT_CLASS: begin
                // Apply sat_add16(sat16(fc_acc >> FB), bias[class]) → logit[class]
                case (fc_class)
                    3'd0: logit0 <= sat_add16(sat16(fc_acc >>> `FRAC_BITS), fc_bias_lane[0]);
                    3'd1: logit1 <= sat_add16(sat16(fc_acc >>> `FRAC_BITS), fc_bias_lane[1]);
                    3'd2: logit2 <= sat_add16(sat16(fc_acc >>> `FRAC_BITS), fc_bias_lane[2]);
                    3'd3: logit3 <= sat_add16(sat16(fc_acc >>> `FRAC_BITS), fc_bias_lane[3]);
                    3'd4: logit4 <= sat_add16(sat16(fc_acc >>> `FRAC_BITS), fc_bias_lane[4]);
                    default: ;
                endcase
                if (fc_class == 3'd4) begin
                    state <= S_FC_FINALIZE;
                end else begin
                    fc_class  <= fc_class + 3'd1;
                    fc_grp_in <= 3'd0;
                    fc_acc    <= 40'sd0;
                    substep   <= 2'd0;
                    state     <= S_FC_MAC;
                end
            end

            S_FC_FINALIZE: begin
                done_fc <= 1'b1;
                state   <= S_FC_DONE;
            end

            S_FC_DONE: begin
                done_phase1 <= 1; done_inception <= 1; done_mamba <= 1; done_all <= 1;
                state       <= S_IDLE;
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
    function signed [15:0] sat16;
        input signed [39:0] v;
        begin
            if      (v >  40'sd32767)  sat16 = 16'sh7FFF;
            else if (v < -40'sd32768)  sat16 = 16'sh8000;
            else                       sat16 = v[15:0];
        end
    endfunction

    // GAP finalize: multiply 24-bit signed sum by INV_T_Q15 (signed 16-bit, e.g. 131
    // for T=250), arithmetic right-shift by 15, saturate to 16-bit signed.
    // Matches Python hw_gap exactly: gap_q = sat16((sum * INV_T_Q15) >> 15).
    function signed [15:0] sat_mul_q15;
        input signed [23:0] sum;
        reg signed [39:0] prod;
        begin
            prod = $signed(sum) * $signed({{16{INV_T_Q15[15]}}, INV_T_Q15});
            sat_mul_q15 = sat16(prod >>> 15);
        end
    endfunction

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

    function [255:0] elem_max16;
        input [255:0] a;
        input [255:0] b;
        integer jj;
        begin
            for (jj = 0; jj < 16; jj = jj + 1)
                elem_max16[jj*16 +: 16] = ($signed(a[jj*16 +: 16]) >= $signed(b[jj*16 +: 16]))
                                          ? a[jj*16 +: 16] : b[jj*16 +: 16];
        end
    endfunction

    function signed [15:0] relu16;
        input signed [15:0] v;
        begin
            relu16 = v[15] ? 16'sd0 : v;
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

    // v2: raw x*x accumulator, no >>>5 pre-shift, no per-channel >>>FRAC_BITS.
    // Each lane x² ≤ 2^30; sum of 16 lanes ≤ 2^34. Return 35-bit (use 40 for alignment).
    function [39:0] norm_sq16_fn;
        input [255:0] word;
        integer       j;
        reg signed [15:0] lane;
        reg signed [31:0] sq;
        reg [39:0]        acc;
        begin
            acc = 40'd0;
            for (j = 0; j < 16; j = j + 1) begin
                lane = $signed(word[j*16 +: 16]);
                sq   = $signed(lane) * $signed(lane);
                acc  = acc + {{8{1'b0}}, $unsigned(sq)};
            end
            norm_sq16_fn = acc;
        end
    endfunction

    // x_norm_fn function removed in CP-1 — RMSNorm normalize now lives in
    // the separate `RMSNorm_Mul` module instance (u_rmsnorm_mul), with a
    // 1-cycle pipeline register between the two saturating multiplies.

endmodule