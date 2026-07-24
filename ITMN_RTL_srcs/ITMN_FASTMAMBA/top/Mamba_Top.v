`include "_parameter.v"

// ============================================================================
// Mamba_Top — Mamba S6 accelerator top-level FSM controller.
//
// ── Iteration dimension key ────────────────────────────────────────────────
//   t      timestep         (0..T_MAX-1)    reg: t_cnt
//   g      output-group     (0..CH_OUT-1 / 0..ch_m_act-1)  reg: ctr_g
//   k      MAC-inner        (0..d_model-1 / 0..d_model/16-1 for RN SQ)  reg: ctr_k
//   l      lane             (0..15)         reg: ctr_l  (M5, M6)
//   r      dt-rank          (0..DT_RANK-1)  reg: ctr_r  (M5)
//   tap    conv tap         (0..3)          reg: tap_cnt (M2 only)
//   slot   M6 load slot     (0..4)          reg: ctr_load (M6 only)
//
// ── Address wire naming: <stage>_<bus>_addr[_<dim><offset>] ────────────────
//   <stage>: mac | rn_sq | rn_ap | m2 | m5 | m6 | m7
//   <bus>:   rd (main RAM read) | wr (main RAM write) | w (weight) | c (const)
//   <dim>:   k (inner offset) | g (group offset) | tap (M2) | r (dt-rank)
//   <offset>: 0/1/2/4 numeric   OR   gnext (next group)   OR   p1 (next counter)
//
// ── Block iteration flow ───────────────────────────────────────────────────
//                 ┌──────────────────────────────────────────────┐
//                 ▼                                              │
//   IDLE → H_INIT → RN → MAC(M1A) → MAC(M1B) → M2 → M3 → MAC(M4) → M5 → M6 → M7 → MAC(M8)
//                                                                                     │
//                                                                          t++, if t<T_MAX
//                                                                                     │
//                                                            else t==T_MAX → DONE ────┘
//
// ── Per-stage state list (numeric naming; steps describe what happens) ─────
//
//   MAC (M1A/M1B/M4/M8): 5 states, cur_stage MUX selects addresses
//     S_MAC1 = PREF        issue read k=0     (main-RAM y-bus)
//     S_MAC2 = WAIT        issue read k=2     (fill BRAM pipe)
//     S_MAC3 = LOAD        issue read k=4     (steady prefetch)
//     S_MAC4 = EXEC (loop k) MAC fire; issue k+4; ctr_k += 2
//     S_MAC5 = LATCH       drain PE; if last g → next stage, else g++, → MAC2
//
//   RN: 13 states — SQ (mean-square) + SCALE + AP (apply)
//     S_RN1  = SQ_PREF     issue x[0] (main RAM)
//     S_RN2  = SQ_WAIT     issue x[1]
//     S_RN3  = SQ_MAC(loop k) MAC x*x fire; issue x[k+2]; ctr_k++
//     S_RN4  = SQ_FINAL    drain 1 cyc (PE latency)
//     S_RN5  = SQ_DONE     capture rsqrt_idx = mean_i_clip
//     S_RN6  = SCALE       S_reg <= rsqrt_data
//     S_RN7  = AP_PREF     issue x[0], g[0] (main + const)
//     S_RN8  = AP_WAIT     BRAM latency wait
//     S_RN9  = AP_MUL1     PE MUL(x,g); wait PE
//     S_RN10 = AP_WAIT1    PE latency
//     S_RN11 = AP_MUL2     PE MUL(x·g, S); prefetch next g if not last
//     S_RN12 = AP_WAIT2    PE latency
//     S_RN13 = AP_WRITE    commit; if last g → S_MAC1 (STG_M1A), else g++, → RN9
//
//   M2 (depthwise conv): 7 states, 4-tap loop
//     S_M2_1 = PREF        issue x[tap0], w[tap0], bias
//     S_M2_2 = WAIT        issue x[tap1], w[tap1]
//     S_M2_3 = TAP(loop tap) MAC; issue tap+2; tap_cnt++; last tap → S_M2_4
//     S_M2_4 = FINAL       drain 1 cyc
//     S_M2_5 = BIAS_ADD    PE ADD (conv + bias)
//     S_M2_6 = BIAS_LATCH  wait PE
//     S_M2_7 = WRITE       commit; if last g → S_M3_1, else g++, → S_M2_2 (fuse prefetch)
//
//   M3 (SiLU stream): 1 state, dual counters (ctr_g read, m3_wr_ptr write, 2-cyc offset)
//     S_M3_1 = STREAM      per-cyc read x_conv[g], write silu[m3_wr_ptr]
//                          exit → S_MAC1 (STG_M4) when m3_wr_ptr == last
//
//   M5 (dt-proj): 9 states, r-dim MAC + bias
//     S_M5_1 = DT_WAIT     wait dt_word BRAM
//     S_M5_2 = DT_LATCH    m5_dt_word_reg <= m_rd_data
//     S_M5_3 = W_PREF      issue W[r=0], bias
//     S_M5_4 = W_WAIT      issue W[r=1]
//     S_M5_5 = MAC(loop r) MAC W·dt fire; issue W[r+2]; ctr_r++
//     S_M5_6 = FINAL       drain
//     S_M5_7 = BIAS_ADD    PE ADD (proj + bias)
//     S_M5_8 = BIAS_LATCH  prefetch next g's W[0]
//     S_M5_9 = WRITE       commit; if last g → S_M6_1, else g++, → S_M5_5 (fuse)
//
//   M6 (SSM scan): 19 states — 5-slot LOAD template + per-lane loop
//     S_M6_1..3   = LOAD_ISSUE/WAIT/LATCH  loop 5×: w0,B,C,delta,u (ctr_load = 0..4)
//     S_M6_4      = DAB_MUL2    PE MUL2(dt,w0) + (dt,B)
//     S_M6_5      = EXP_WAIT    wait Exp_LUT
//     S_M6_6      = DAB_LATCH   capture m6_dA, m6_dB
//     S_M6_7      = SSM_MUL     PE MUL(dA, h_old)
//     S_M6_8      = SSM_WAIT    PE latency
//     S_M6_9      = ADD_ISSUE   PE ADD (dA·h + dB·x → h_new)
//     S_M6_10     = SSM_LATCH   write h_new to H_RegFile
//     S_M6_11     = Y_MAC       PE MAC (y += h_new · C)
//     S_M6_12/13  = Y_WAIT/LATCH  capture y_ch
//     S_M6_14/15/16 = DU_MUL/WAIT/LATCH  PE MUL(D, u)
//     S_M6_17/18  = YSUM_WAIT/LATCH  PE ADD (y_ch + D·u)  [Phase 3.6 fix]
//                   next lane (ctr_l++) → S_M6_4  |  last lane → S_M6_19
//     S_M6_19     = WRITE_GRP   commit y_ssm[g]; if last g → S_M7_1
//                              else g++, ctr_load=3, → S_M6_1 (re-entry only loads δ,u)
//
//   M7 (gate mul): 7 states
//     S_M7_1 = Z_PREF      issue z[g], y_ssm[g]
//     S_M7_2 = Z_WAIT      BRAM latency
//     S_M7_3 = Z_LATCH     silu_z_reg <= silu(z_word)
//     S_M7_4 = Y_WAIT      PE latency
//     S_M7_5 = MUL         PE MUL (y_ssm · silu(z))
//     S_M7_6 = LATCH       PE latency
//     S_M7_7 = WRITE       commit; if last g → S_MAC1 (STG_M8), else g++, → S_M7_3 (fuse)
//
// ── Verb glossary (used in comments only, not in state names anymore) ─────
//   PREF/ISSUE  drive first BRAM read address
//   WAIT        hold 1 cyc for BRAM 2-cyc read latency OR PE 1-cyc latency
//   LOAD        issue additional BRAM read mid-pipeline
//   MAC/MUL/ADD PE fire — fabric MAC/multiply/add through Mamba_PE
//   LATCH       capture BRAM data or PE output into internal reg
//   FINAL       drain 1 cyc after last inner (PE pipeline flush)
//   WRITE       commit result to main-RAM; branch group-inc vs stage-transition
// ============================================================================
module Mamba_Top (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    output reg          done_stage,
    output reg          done_all,

    input  wire [3:0]   run_stage,
    input  wire [9:0]   T_MAX,
    input  wire [6:0]   CH_OUT,      // Widened for Mamba2 (d_model up to 2048 → 128 grp)
    input  wire [6:0]   CH_M,        // Widened for Mamba2 (d_inner up to 2048 → 128 grp)
    input  wire [3:0]   DT_RANK,
    input  wire [3:0]   N_STATE_GRP, // d_state / 16 (ITMN default = 1; Mamba2 = 8)
    input  wire         USE_M5,      // 1 = run M5 dt-proj (ITMN); 0 = skip (Mamba2, dt from in-proj)
    input  wire [4:0]   XP_OUT_GRP_IN, // M4 x_proj output groups. ITMN Mamba1 = 3 (dt+B+C = 48 elems);
                                       // Mamba2 = 18 (dt=24 + B=128 + C=128 = 280 elems)

    input  wire         dma_write_en,
    input  wire [1:0]   dma_target,
    input  wire [14:0]  dma_addr,
    input  wire [255:0] dma_wdata,

    input  wire         dma_read_en,
    input  wire [1:0]   dma_rtarget,
    input  wire [14:0]  dma_raddr,
    output wire [255:0] dma_rdata
);

    wire [7:0]  ch_m_act = (CH_M == 7'd0) ? 8'd128 : {1'b0, CH_M};
    wire [11:0] d_inner  = {ch_m_act, 4'b0};   // Widened: up to 2048
    // H init limit = d_inner * N_STATE_GRP (Mamba1=1, Mamba2=8)
    wire [15:0] h_init_limit = (N_STATE_GRP == 4'd8) ? {1'b0, d_inner, 3'b0} :
                                                       {4'b0, d_inner};
    wire [10:0] d_model  = {CH_OUT, 4'b0};     // Widened: up to 2032
    wire [9:0]  t_last   = T_MAX - 10'd1;
    // Generalized log2 for d_model (ITMN B0-B4 + Mamba2 d_model=768 → log2=9)
    wire [3:0]  log2_dm  =
        (CH_OUT == 7'd4)   ? 4'd2  :   // d_model=64  (B0/B4 style)
        (CH_OUT == 7'd8)   ? 4'd3  :   // d_model=128
        (CH_OUT == 7'd16)  ? 4'd4  :   // d_model=256
        (CH_OUT == 7'd32)  ? 4'd5  :   // d_model=512
        (CH_OUT == 7'd48)  ? 4'd9  :   // d_model=768 (Mamba2-130M) — log2(768)=9.58→9
                             4'd10;    // d_model=1024+
    wire [5:0]  rn_total_shift = {2'b0, log2_dm} + 6'd19;
    wire       [4:0] XP_OUT_GRP = (XP_OUT_GRP_IN == 5'd0) ? 5'd3 : XP_OUT_GRP_IN;

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

    // Stage enum mirrors of `_parameter.v macros (14:0-safe for cur_stage compares)
    localparam [3:0] STG_M1A = `STG_M1A;
    localparam [3:0] STG_M1B = `STG_M1B;
    localparam [3:0] STG_M2  = `STG_M2;
    localparam [3:0] STG_M3  = `STG_M3;
    localparam [3:0] STG_M4  = `STG_M4;
    localparam [3:0] STG_M5  = `STG_M5;
    localparam [3:0] STG_M6  = `STG_M6;
    localparam [3:0] STG_M7  = `STG_M7;
    localparam [3:0] STG_M8  = `STG_M8;
    localparam [3:0] STG_RN  = `STG_RN;

    localparam [6:0] S_IDLE          = 7'd0;
    localparam [6:0] S_H_INIT        = 7'd1;
    // Shared MAC pipeline (M1A/M1B/M4/M8): PREF → WAIT → LOAD → EXEC → LATCH
    localparam [6:0] S_MAC1      = 7'd2;
    localparam [6:0] S_MAC2      = 7'd3;
    localparam [6:0] S_MAC4      = 7'd4;
    localparam [6:0] S_RN1    = 7'd6;
    localparam [6:0] S_RN2    = 7'd7;
    localparam [6:0] S_RN3     = 7'd8;
    localparam [6:0] S_RN4   = 7'd9;
    localparam [6:0] S_RN5    = 7'd10;
    // S_RN_RSQ_WAIT (7'd11) removed — RSqrt_ROM combinational, S_LATCH samples
    // rsqrt_data directly cyc after SQ_DONE (idx_r registered at SQ_DONE edge).
    localparam [6:0] S_RN6    = 7'd12;
    localparam [6:0] S_RN7    = 7'd13;
    localparam [6:0] S_RN8    = 7'd14;
    localparam [6:0] S_RN9    = 7'd15;
    localparam [6:0] S_RN10   = 7'd16;
    localparam [6:0] S_RN11    = 7'd17;
    localparam [6:0] S_RN12   = 7'd18;
    localparam [6:0] S_RN13   = 7'd19;
    localparam [6:0] S_M2_1       = 7'd21;
    localparam [6:0] S_M2_2       = 7'd22;
    localparam [6:0] S_M2_3        = 7'd23;
    localparam [6:0] S_M2_4      = 7'd24;   // drain cyc after tap 3 (PE latency)
    // S_M2_BIAS_PREF (25) and S_M2_BIAS_WAIT (26) removed: bias addr prefetched in S_M2_1
    localparam [6:0] S_M2_5   = 7'd27;
    localparam [6:0] S_M2_6 = 7'd28;
    localparam [6:0] S_M2_7      = 7'd29;
    localparam [6:0] S_M3_1     = 7'd31;
    localparam [6:0] S_M5_1    = 7'd36;
    localparam [6:0] S_M5_2   = 7'd37;
    localparam [6:0] S_M5_3     = 7'd38;
    localparam [6:0] S_M5_4     = 7'd39;
    localparam [6:0] S_M5_5        = 7'd40;
    localparam [6:0] S_M5_6      = 7'd41;
    // S_M5_BIAS_PREF (42) and S_M5_BIAS_WAIT (43) removed: bias addr prefetched in S_M5_3
    localparam [6:0] S_M5_7   = 7'd44;
    localparam [6:0] S_M5_8 = 7'd45;
    localparam [6:0] S_M5_9      = 7'd46;
    localparam [6:0] S_M7_1     = 7'd48;
    localparam [6:0] S_M7_2     = 7'd49;
    localparam [6:0] S_M7_3    = 7'd50;
    localparam [6:0] S_M7_4     = 7'd52;
    localparam [6:0] S_M7_5        = 7'd53;
    localparam [6:0] S_M7_6      = 7'd54;
    localparam [6:0] S_M7_7      = 7'd55;
    localparam [6:0] S_MAC5        = 7'd98;
    localparam [6:0] S_M6_8      = 7'd99;
    localparam [6:0] S_M6_15       = 7'd101;
    localparam [6:0] S_MAC3         = 7'd102;
    // Collapsed M6 load template (replaces 9 states:
    //   LOAD_W0_PREF/WAIT/LATCH + LOAD_B_LATCH + LOAD_C_LATCH +
    //   LOAD_DT_PREF/WAIT/LATCH + LOAD_U_LATCH)
    // ctr_load selects slot 0..4 (w0,w1,w2,delta,u); side-issues (D const,
    // A weight, H) piggyback at slot-1 latch for 2-cyc alignment.
    // First entry (from M5 WRITE): ctr_load starts at 0, does all 5 slots (7 cyc).
    // Re-entry (from M6 WRITE_GRP): ctr_load starts at 3, only slots 3..4 (4 cyc).
    localparam [6:0] S_M6_1 = 7'd58;
    localparam [6:0] S_M6_2  = 7'd59;
    localparam [6:0] S_M6_3 = 7'd60;
    localparam [6:0] S_M6_4      = 7'd74;
    localparam [6:0] S_M6_5      = 7'd75;
    localparam [6:0] S_M6_6     = 7'd76;
    localparam [6:0] S_M6_7       = 7'd80;
    localparam [6:0] S_M6_10     = 7'd81;
    localparam [6:0] S_M6_9     = 7'd82;
    localparam [6:0] S_M6_11         = 7'd86;
    localparam [6:0] S_M6_12        = 7'd87;
    localparam [6:0] S_M6_13       = 7'd88;
    localparam [6:0] S_M6_14        = 7'd89;
    localparam [6:0] S_M6_16      = 7'd90;
    // Phase 3.6: y_ch + du addition moved into PE ADD mode (was combinational
    // 17-bit add + sat outside PE). Sequence: DU_LATCH fires PE ADD →
    // YSUM_WAIT (PE 1-cyc latency) → YSUM_LATCH stores sat16(y_ch+du) result.
    localparam [6:0] S_M6_17     = 7'd91;
    localparam [6:0] S_M6_18    = 7'd92;
    localparam [6:0] S_M6_19     = 7'd93;
    localparam [6:0] S_DONE             = 7'd127;

    reg [6:0]   state;
    reg [3:0]   cur_stage;
    reg [9:0]   t_cnt;
    reg [6:0]   ctr_g;              // Widened 5→7-bit for Mamba2 (up to 96 grp)
    reg [10:0]  ctr_k;               // Widened 9→11-bit (up to d_inner=1536)
    reg [1:0]   tap_cnt;
    reg [4:0]   ctr_r;
    reg [6:0]   m3_wr_ptr;              // Widened 5→7-bit for Mamba2 CH_M=96
    reg [2:0]   ctr_load;              // Phase 1.5: M6 load-template slot counter
    reg [2:0]   ctr_s;                 // NEW: M6 state-group counter (0..N_STATE_GRP-1) for Mamba2 d_state=128
    reg [255:0] silu_z_reg;
    reg [255:0] m5_dt_word_reg;
    reg [3:0]   ctr_l;              // lane counter (cross-stage: M5 broadcast, M6 SSM per-lane)
    reg [15:0]  S_reg;
    reg [13:0]  h_init_cnt;           // Widened 9→14-bit for expanded H_RegFile (up to 16384)

    // Mamba2 remap: w0=B[s], w1=C[s], w2=A[s]. Barrel-shift dropped (Mamba1-only).
    reg [255:0] m6_delta_word, m6_u_word, m6_w0_reg, m6_w1_reg, m6_w2_reg, m6_D_word;
    reg [255:0] m6_dA_reg, m6_dB_reg, m6_ssm_grp_acc;
    reg signed [23:0] m6_y_ch_reg;  // Widened for 8-s external accumulation

    reg  [14:0]  m_rd_addr;
    reg          m_we;
    reg  [14:0]  m_wr_addr;
    reg  [255:0] m_wr_data;
    reg  [14:0]  w_rd_addr, w_rd_addr2;
    wire [255:0] m_rd_data, w_rd_data, w_rd_data2;

    reg  [2:0]                cl_op_mode;
    reg                       cl_clear_acc;
    reg  [16*`DATA_W-1:0]     cl_in_W1_vec, cl_in_H_ext, cl_in_W2_vec, cl_in_X_vec;
    wire [16*`DATA_W-1:0]     cl_out_vec, cl_out_next_vec;
    wire [16*`DATA_W-1:0]     cl_out_vec2, cl_out_next_vec2;
    wire [16*`ACC_W-1:0]      cl_acc_raw_vec;
    wire [`DATA_W+4:0]        cl_y_reduce_out;

    wire [3:0]   y_lane_sel  = ctr_k[3:0];       // Lane within word (16 lanes)
    wire [15:0]  y_scalar    = m_rd_data[y_lane_sel*16 +: 16];
    wire [255:0] y_broadcast = {16{y_scalar}};
    wire [3:0]   y_lane_sel2 = ctr_k[3:0] + 4'd1;
    wire [15:0]  y_scalar2   = m_rd_data[y_lane_sel2*16 +: 16];
    wire [255:0] y_broadcast2 = {16{y_scalar2}};

    reg  [14:0]  const_rd_addr_r;
    wire [255:0] const_rd_data;
    reg  [12:0]  rsqrt_idx_r;
    wire [15:0]  rsqrt_data;
    wire [255:0] silu_out_w, sp_out_w, exp_out_w;

    wire [255:0] silu_in_drv, sp_in_drv, exp_in_drv;

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
        .const_read_addr  (const_rd_addr_r),
        .const_read_data  (const_rd_data),
        .dma_write_en     (dma_write_en),
        .dma_target       (dma_target),
        .dma_addr         (dma_addr),
        .dma_wdata        (dma_wdata),
        .dma_read_en      (dma_read_en),
        .dma_rtarget      (dma_rtarget),
        .dma_raddr        (dma_raddr),
        .dma_rdata        (dma_rdata)
    );

    LUT_Bank u_lut (
        .silu_in_flat  (silu_in_drv),
        .sp_in_flat    (sp_in_drv),
        .exp_in_flat   (exp_in_drv),
        .silu_out_flat (silu_out_w),
        .sp_out_flat   (sp_out_w),
        .exp_out_flat  (exp_out_w),
        .rsqrt_idx     (rsqrt_idx_r),
        .rsqrt_data    (rsqrt_data)
    );

    reg                       m6_h_from_rf_r;
    reg  [13:0]               m6_h_rd_addr_r;    // Widened 9→14 bit for H depth up to 16384
    reg                       m6_h_wr_en_r;
    reg  [13:0]               m6_h_wr_addr_r;    // Widened 9→14 bit
    reg                       m6_h_wr_from_pe_r;
    reg  [16*`DATA_W-1:0]     m6_h_wr_data_ext_r;

    wire h_ctrl_active = (cur_stage == STG_M6) || (state == S_H_INIT);
    wire                      cl_h_from_rf_w     = h_ctrl_active ? m6_h_from_rf_r     : 1'b0;
    wire [13:0]               cl_h_rd_addr_w     = h_ctrl_active ? m6_h_rd_addr_r     : 14'b0;
    wire                      cl_h_wr_en_w       = h_ctrl_active ? m6_h_wr_en_r       : 1'b0;
    wire [13:0]               cl_h_wr_addr_w     = h_ctrl_active ? m6_h_wr_addr_r     : 14'b0;
    wire                      cl_h_wr_from_pe_w  = h_ctrl_active ? m6_h_wr_from_pe_r  : 1'b0;
    wire [16*`DATA_W-1:0]     cl_h_wr_data_ext_w = h_ctrl_active ? m6_h_wr_data_ext_r : 256'b0;

    M_Cluster #(.H_ADDR_W(14), .H_DEPTH(16384)) u_mc (
        .clk          (clk),
        .rst          (rst),
        .op_mode      (cl_op_mode),
        .clear_acc    (cl_clear_acc),
        .in_W1_vec    (cl_in_W1_vec),
        .in_H_ext     (cl_in_H_ext),
        .in_W2_vec    (cl_in_W2_vec),
        .in_X_vec     (cl_in_X_vec),
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

    wire signed [`ACC_W+4:0] mean_i_signed = sum_d_wide >>> rn_total_shift;
    wire [12:0] mean_i_clip =
        (mean_i_signed < 0)         ? 13'd0    :
        (mean_i_signed > 45'sd8191) ? 13'd8191 :
                                       mean_i_signed[12:0];
    wire [255:0] S_broadcast = {16{S_reg}};

    assign silu_in_drv = (state == S_M3_1 ) ? m_rd_data :
                         (state == S_M7_3) ? m_rd_data :
                                                    256'b0;
    assign sp_in_drv   = (state == S_M5_9  ) ? cl_out_vec : 256'b0;
    assign exp_in_drv  = (state == S_M6_6) ? cl_out_vec : 256'b0;

    wire [11:0] mac_len =
        (cur_stage == STG_M4) ? d_inner :
        (cur_stage == STG_M8) ? d_inner :
                              {1'b0, d_model};
    wire [7:0]  mac_grp_count =
        (cur_stage == STG_M8) ? {1'b0, CH_OUT} :
        (cur_stage == STG_M4) ? {3'b0, XP_OUT_GRP} :
                              ch_m_act;
    wire [14:0] mac_in_base =
        (cur_stage == STG_M4) ? PT_U :
        (cur_stage == STG_M8) ? PT_Y_GATED :
                              PT_X_NORM;
    // OUTPROJ permanent at W_OUTPROJ_BASE (post-refactor: streaming removed,
    // all weights fit permanently in expanded ram_weight).
    wire [14:0] mac_w_base =
        (cur_stage == STG_M1A) ? W_INPROJ_X_BASE :
        (cur_stage == STG_M1B) ? W_INPROJ_Z_BASE :
        (cur_stage == STG_M4) ? W_XPROJ_BASE :
                              W_OUTPROJ_BASE;

    wire [14:0] m1a_wr_addr = PT_X_INNER_CIRC
                              + ({13'b0, t_cnt[1:0]} * {7'b0, ch_m_act})
                              + {8'b0, ctr_g};
    wire [14:0] m1b_wr_addr = PT_Z_GATE + {8'b0, ctr_g};
    wire [14:0] m4_wr_addr  = PT_X_PROJ + {8'b0, ctr_g};
    wire [14:0] m8_wr_addr  = PT_MAMBA_OUT
                              + (t_cnt * {8'b0, CH_OUT})
                              + {8'b0, ctr_g};
    wire [14:0] mac_wr_addr =
        (cur_stage == STG_M1A) ? m1a_wr_addr :
        (cur_stage == STG_M1B) ? m1b_wr_addr :
        (cur_stage == STG_M4) ? m4_wr_addr  :
                              m8_wr_addr;

    wire [7:0]  mac_grp_last = mac_grp_count - 8'd1;
    wire [14:0] mac_len_ext  = {3'b0, mac_len};
    wire [14:0] w_grp_base   = mac_w_base + ({8'b0, ctr_g} * mac_len_ext);

    wire [10:0] ctr_k_p1 = ctr_k + 11'd1;
    wire [10:0] ctr_k_p2 = ctr_k + 11'd2;
    wire [10:0] ctr_k_p4 = ctr_k + 11'd4;

    // ctr_k high bits = word index (per-lane grp of 16), so [10:4] used as addr offset
    wire [14:0] mac_rd_addr_k0 = mac_in_base + {8'b0, ctr_k[10:4]};
    wire [14:0] mac_rd_addr_k2  = mac_in_base + {8'b0, ctr_k_p2[10:4]};
    wire [14:0] mac_rd_addr_k4  = mac_in_base + {8'b0, ctr_k_p4[10:4]};
    wire [14:0] mac_w_addr_k0 = w_grp_base + {4'b0, ctr_k};
    wire [14:0] mac_w_addr_k2  = w_grp_base + {4'b0, ctr_k_p2};
    wire [14:0] mac_w_addr_k4  = w_grp_base + {4'b0, ctr_k_p4};

    wire [6:0]  ctr_g_p1 = ctr_g + 7'd1;
    wire [14:0] w_grp_base_next = mac_w_base + ({8'b0, ctr_g_p1} * mac_len_ext);
    wire [14:0] mac_w_addr_gnext = w_grp_base_next;
    wire [14:0] mac_rd_addr_g0 = mac_in_base;

    wire [7:0]  inner_grp_last = ch_m_act - 8'd1;

    wire [7:0]  rn_x_per_t     = {1'b0, CH_OUT};
    wire [7:0]  rn_x_grp_last  = rn_x_per_t - 8'd1;
    wire [14:0] rn_x_per_t_ext = {7'b0, rn_x_per_t};
    wire [14:0] rn_t_offset_x  = t_cnt * rn_x_per_t_ext;
    wire [14:0] rn_ap_rd_addr_g0  = PT_INPUT + rn_t_offset_x + {8'b0, ctr_g};
    wire [14:0] rn_ap_rd_addr_g1   = PT_INPUT + rn_t_offset_x + {8'b0, ctr_g_p1};
    wire [14:0] rn_ap_c_addr_g0  = C_W_NORM_BASE + {8'b0, ctr_g};
    wire [14:0] rn_ap_c_addr_g1   = C_W_NORM_BASE + {8'b0, ctr_g_p1};
    wire [14:0] rn_wr_addr     = PT_X_NORM + {8'b0, ctr_g};
    // RN SQ iterates ctr_k over d_model/16 groups. Use ctr_k[6:0] (128 grp max).
    wire [14:0] rn_sq_rd_addr_k0  = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k[6:0]};
    wire [14:0] rn_sq_rd_addr_k1   = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k_p1[6:0]};
    wire [14:0] rn_sq_rd_addr_k2   = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k_p2[6:0]};
    wire [10:0] rn_sq_last     = {3'b0, rn_x_grp_last};

    wire signed [10:0] eff_t_for_tap = {1'b0, t_cnt} - 11'd3 + {9'b0, tap_cnt};
    wire        m2_pad     = (eff_t_for_tap < 0);
    wire [1:0]  eff_t_slot = eff_t_for_tap[1:0];
    wire [14:0] m2_rd_addr_tap0 = PT_X_INNER_CIRC
                                    + ({13'b0, eff_t_slot} * {7'b0, ch_m_act})
                                    + {8'b0, ctr_g};

    wire [2:0]  tap_p1 = {1'b0, tap_cnt} + 3'd1;
    wire [2:0]  tap_p2 = {1'b0, tap_cnt} + 3'd2;
    wire signed [11:0] eff_t_p1 = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p1});
    wire signed [11:0] eff_t_p2 = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p2});
    wire [1:0]  eff_t_slot_p1 = eff_t_p1[1:0];
    wire [1:0]  eff_t_slot_p2 = eff_t_p2[1:0];
    wire [14:0] m2_rd_addr_tap1 = PT_X_INNER_CIRC
                               + ({13'b0, eff_t_slot_p1} * {7'b0, ch_m_act})
                               + {8'b0, ctr_g};
    wire [14:0] m2_rd_addr_tap2 = PT_X_INNER_CIRC
                               + ({13'b0, eff_t_slot_p2} * {7'b0, ch_m_act})
                               + {8'b0, ctr_g};

    wire [14:0] m2_w_addr_tap0 = W_DW_BASE + ({7'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {13'b0, tap_cnt};
    wire [14:0] m2_w_addr_tap1      = W_DW_BASE + ({7'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {12'b0, tap_p1};
    wire [14:0] m2_w_addr_tap2      = W_DW_BASE + ({7'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {12'b0, tap_p2};
    wire [14:0] m2_c_addr      = C_B_DW_BASE + {8'b0, ctr_g};
    wire [14:0] m2_wr_addr        = PT_X_CONV + {8'b0, ctr_g};

    // M2 next-group tap-0 prefetch (for WRITE→WAIT state fusion, saves 1 cyc/group boundary)
    wire signed [10:0] eff_t_tap0      = {1'b0, t_cnt} - 11'd3;
    wire [1:0]         eff_t_slot_tap0 = eff_t_tap0[1:0];
    wire [14:0] m2_rd_addr_gnext = PT_X_INNER_CIRC
                                     + ({13'b0, eff_t_slot_tap0} * {7'b0, ch_m_act})
                                     + {8'b0, ctr_g_p1};
    wire [14:0] m2_w_addr_gnext = W_DW_BASE + ({7'b0, ctr_g_p1} * `M2_WEIGHT_STRIDE);
    wire [14:0] m2_c_addr_gnext  = C_B_DW_BASE + {8'b0, ctr_g_p1};

    wire [14:0]  m5_dt_addr      = PT_X_PROJ + 15'd0;
    wire [15:0]  m5_dt_scalar    = m5_dt_word_reg[ctr_l*16 +: 16];
    wire [255:0] m5_dt_broadcast = {16{m5_dt_scalar}};
    wire [4:0]   dt_rank_last    = {1'b0, DT_RANK - 4'd1};
    wire [4:0]   ctr_r_p1         = ctr_r + 5'd1;
    wire [4:0]   ctr_r_p2         = ctr_r + 5'd2;
    wire [14:0]  m5_w_addr_r0   = W_DTPROJ_BASE + ({8'b0, ctr_g} * {11'b0, DT_RANK}) + {10'b0, ctr_r};
    wire [14:0]  m5_w_addr_r1    = W_DTPROJ_BASE + ({8'b0, ctr_g} * {11'b0, DT_RANK}) + {10'b0, ctr_r_p1};
    wire [14:0]  m5_w_addr_r2    = W_DTPROJ_BASE + ({8'b0, ctr_g} * {11'b0, DT_RANK}) + {10'b0, ctr_r_p2};
    wire [14:0]  m5_c_addr    = C_B_DT_BASE + {8'b0, ctr_g};
    wire [14:0]  m5_wr_addr      = PT_DELTA + {8'b0, ctr_g};
    // Next-group prefetch helpers (for WRITE→MAC fusion, skip W_PREF/W_WAIT)
    wire [14:0]  m5_w_addr_gnext_r0 = W_DTPROJ_BASE + ({8'b0, ctr_g_p1} * {11'b0, DT_RANK});
    wire [14:0]  m5_w_addr_gnext_r1 = m5_w_addr_gnext_r0 + 15'd1;
    wire [14:0]  m5_c_addr_gnext = C_B_DT_BASE + {8'b0, ctr_g_p1};

    wire [14:0] m7_z_addr    = PT_Z_GATE + {8'b0, ctr_g};
    wire [14:0] m7_z_addr_p1 = PT_Z_GATE + {8'b0, ctr_g} + 15'd1;
    wire [14:0] m7_y_addr    = PT_Y_SSM  + {8'b0, ctr_g};
    wire [14:0] m7_wr_addr   = PT_Y_GATED + {8'b0, ctr_g};

    wire [15:0]  m6_dt_scalar    = m6_delta_word[ctr_l*16 +: 16];
    wire [255:0] m6_dt_broadcast = {16{m6_dt_scalar}};
    wire [15:0]  m6_u_scalar     = m6_u_word[ctr_l*16 +: 16];
    wire [255:0] m6_u_broadcast  = {16{m6_u_scalar}};
    wire [15:0]  m6_D_scalar     = m6_D_word[ctr_l*16 +: 16];
    wire [255:0] m6_D_broadcast  = {16{m6_D_scalar}};

    // H_RegFile address: {ctr_g[6:0], ctr_l[3:0], ctr_s[2:0]} = 14-bit
    //   ctr_s = state-group index (0..N_STATE_GRP-1) — inner-most for locality
    //   ctr_l = lane within word (0..15)
    //   ctr_g = channel-group (0..CH_M-1)
    wire [13:0]  m6_c = {ctr_g, ctr_l, ctr_s};

    // Mamba2 per-s addresses (s = ctr_s, N_STATE_GRP=8)
    wire [10:0]  m6_channel      = {ctr_g, ctr_l};
    wire [14:0]  m6_B_addr_s     = PT_X_PROJ + {12'b0, ctr_s};
    wire [14:0]  m6_C_addr_s     = PT_X_PROJ + {7'b0, `N_STATE_GRP} + {12'b0, ctr_s};
    wire [14:0]  m6_A_addr_s     = W_A_BASE  + {1'b0, m6_channel, 3'b0} + {12'b0, ctr_s};
    wire [14:0]  m6_delta_addr   = PT_DELTA + {8'b0, ctr_g};
    wire [14:0]  m6_u_addr       = PT_U     + {8'b0, ctr_g};
    wire [14:0]  m6_D_addr_const = C_D_PARAM_BASE + {8'b0, ctr_g};
    wire [14:0]  m6_y_ssm_wr_addr = PT_Y_SSM + {8'b0, ctr_g};

    wire signed [`ACC_W+4:0] m6_y_ch_full = sum_d_wide >>> `FRAC_BITS;
    wire signed [15:0] m6_y_ch_sat =
        (m6_y_ch_full >  45'sd32767)  ? 16'sh7FFF :
        (m6_y_ch_full < -45'sd32768)  ? 16'sh8000 :
                                         m6_y_ch_full[15:0];
    // 24→16 sat for m6_y_ch_reg accumulator (used at YSUM_ADD)
    wire signed [15:0] m6_y_ch_reg_sat =
        (m6_y_ch_reg >  24'sd32767)  ? 16'sh7FFF :
        (m6_y_ch_reg < -24'sd32768)  ? 16'sh8000 :
                                        m6_y_ch_reg[15:0];
    // Phase 3.6: m6_ysum comb-add+sat removed. y_ch + du now computed by
    // PE ADD (see S_M6_16 / S_M6_17 / S_M6_18).
    // Store reads cl_out_vec[15:0] directly at YSUM_LATCH.

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state              <= S_IDLE;
            cur_stage          <= 4'd0;
            t_cnt              <= 10'd0;
            ctr_g              <= 7'd0;
            ctr_k              <= 11'd0;
            tap_cnt            <= 2'd0;
            ctr_r              <= 5'd0;
            m3_wr_ptr          <= 7'd0;
            ctr_l              <= 4'd0;
            ctr_load           <= 3'd0;
            ctr_s              <= 3'd0;
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
            h_init_cnt         <= 14'd0;
            ctr_l            <= 4'd0;
            m6_delta_word      <= 256'd0;
            m6_u_word          <= 256'd0;
            m6_w0_reg          <= 256'd0;
            m6_w1_reg          <= 256'd0;
            m6_w2_reg          <= 256'd0;
            m6_D_word          <= 256'd0;
            m6_dA_reg          <= 256'd0;
            m6_dB_reg          <= 256'd0;
            m6_ssm_grp_acc     <= 256'd0;
            m6_y_ch_reg        <= 24'sd0;
            m6_h_from_rf_r     <= 1'b0;
            m6_h_rd_addr_r     <= 14'd0;
            m6_h_wr_en_r       <= 1'b0;
            m6_h_wr_addr_r     <= 14'd0;
            m6_h_wr_from_pe_r  <= 1'b0;
            m6_h_wr_data_ext_r <= 256'd0;
            done_stage         <= 1'b0;
            done_all           <= 1'b0;
        end else begin
            cl_op_mode           <= `MAMBA_PE_IDLE;
            cl_clear_acc         <= 1'b0;
            m_we                 <= 1'b0;
            m6_h_wr_en_r         <= 1'b0;

            case (state)
                S_IDLE: begin
                    done_stage <= 1'b0;
                    done_all   <= 1'b0;
                    if (start) begin
                        t_cnt      <= 10'd0;
                        ctr_g      <= 7'd0;
                        ctr_k      <= 11'd0;
                        tap_cnt    <= 2'd0;
                        ctr_r      <= 5'd0;
                        ctr_l      <= 4'd0;
                        ctr_s      <= 3'd0;
                        h_init_cnt <= 14'd0;
                        cur_stage  <= STG_M6;
                        state      <= S_H_INIT;
                    end
                end

                S_H_INIT: begin
                    if ({2'b0, h_init_cnt} < h_init_limit) begin
                        m6_h_wr_en_r       <= 1'b1;
                        m6_h_wr_addr_r     <= h_init_cnt;
                        m6_h_wr_from_pe_r  <= 1'b0;
                        m6_h_wr_data_ext_r <= 256'd0;
                        h_init_cnt         <= h_init_cnt + 14'd1;
                    end else begin
                        h_init_cnt <= 14'd0;
                        cur_stage  <= STG_RN;
                        state      <= S_RN1;
                    end
                end

                S_MAC1: begin
                    m_rd_addr  <= mac_rd_addr_k0;
                    w_rd_addr  <= mac_w_addr_k0;
                    w_rd_addr2 <= mac_w_addr_k0 + 15'd1;
                    state      <= S_MAC2;
                end
                S_MAC2: begin
                    w_rd_addr    <= mac_w_addr_k2;
                    w_rd_addr2   <= mac_w_addr_k2 + 15'd1;
                    m_rd_addr    <= mac_rd_addr_k2;
                    ctr_k     <= 11'd0;
                    state        <= S_MAC3;
                end
                S_MAC3: begin
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_W2_vec <= w_rd_data2;
                    cl_in_H_ext  <= y_broadcast;
                    cl_in_X_vec  <= y_broadcast2;
                    cl_op_mode   <= `MAMBA_PE_MAC2;
                    cl_clear_acc <= 1'b1;
                    m_rd_addr    <= mac_rd_addr_k4;
                    w_rd_addr    <= mac_w_addr_k4;
                    w_rd_addr2   <= mac_w_addr_k4 + 15'd1;
                    ctr_k     <= ctr_k_p2;
                    state        <= S_MAC4;
                end
                S_MAC4: begin
                    cl_op_mode <= `MAMBA_PE_MAC2;
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_W2_vec <= w_rd_data2;
                    cl_in_H_ext  <= y_broadcast;
                    cl_in_X_vec  <= y_broadcast2;
                    m_rd_addr  <= mac_rd_addr_k4;
                    w_rd_addr  <= mac_w_addr_k4;
                    w_rd_addr2 <= mac_w_addr_k4 + 15'd1;
                    ctr_k   <= ctr_k_p2;
                    if (ctr_k == mac_len) begin
                        state <= S_MAC5;
                    end
                end
                S_MAC5: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= mac_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (ctr_g == mac_grp_last) begin
                        ctr_g <= 7'd0;
                        ctr_k  <= 11'd0;
                        case (cur_stage)
                            STG_M1A: begin cur_stage <= STG_M1B; state <= S_MAC1; end
                            STG_M1B: begin cur_stage <= STG_M2;  state <= S_M2_1;  end
                            STG_M4: begin
                                // Prefetch dt_raw during LATCH (port A used for write, port B free)
                                // → skip S_M5_DT_READ. Keep DT_WAIT: BRAM needs 1-cyc read latency
                                // between addr-issue and dout-valid.
                                // FastMamba port: USE_M5=0 → skip M5 (Mamba2 has no dt-proj)
                                if (USE_M5) begin
                                    cur_stage <= STG_M5;
                                    m_rd_addr <= m5_dt_addr;
                                    state     <= S_M5_1;
                                end else begin
                                    cur_stage <= STG_M6;
                                    ctr_l     <= 4'd0;
                                    ctr_s     <= 3'd0;
                                    ctr_load  <= 3'd0;
                                    state     <= S_M6_1;
                                end
                            end
                            STG_M8: begin
                                if (t_cnt == t_last) begin
                                    state <= S_DONE;
                                end else begin
                                    t_cnt     <= t_cnt + 10'd1;
                                    cur_stage <= STG_RN;
                                    state     <= S_RN1;
                                end
                            end
                            default: state <= S_IDLE;
                        endcase
                    end else begin
                        ctr_g  <= ctr_g_p1;
                        ctr_k   <= 11'd0;
                        m_rd_addr  <= mac_rd_addr_g0;
                        w_rd_addr  <= mac_w_addr_gnext;
                        w_rd_addr2 <= mac_w_addr_gnext + 15'd1;
                        state      <= S_MAC2;
                    end
                end

                S_RN1: begin
                    ctr_k  <= 11'd0;
                    m_rd_addr <= rn_sq_rd_addr_k0;
                    state     <= S_RN2;
                end
                S_RN2: begin
                    m_rd_addr <= rn_sq_rd_addr_k1;
                    state     <= S_RN3;
                end
                S_RN3: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (ctr_k == 11'd0);
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= m_rd_data;
                    m_rd_addr    <= rn_sq_rd_addr_k2;
                    ctr_k     <= ctr_k_p1;
                    if (ctr_k == rn_sq_last) state <= S_RN4;
                end
                S_RN4: state <= S_RN5;
                S_RN5: begin
                    rsqrt_idx_r <= mean_i_clip;
                    state       <= S_RN6;
                end
                S_RN6: begin
                    S_reg  <= rsqrt_data;
                    ctr_g <= 7'd0;
                    state  <= S_RN7;
                end
                S_RN7: begin
                    m_rd_addr       <= rn_ap_rd_addr_g0;
                    const_rd_addr_r <= rn_ap_c_addr_g0;
                    state           <= S_RN8;
                end
                S_RN8: state <= S_RN9;
                S_RN9: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= const_rd_data;
                    state        <= S_RN10;
                end
                S_RN10: state <= S_RN11;
                S_RN11: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= cl_out_vec;
                    cl_in_H_ext  <= S_broadcast;
                    if (ctr_g != rn_x_grp_last) begin
                        m_rd_addr       <= rn_ap_rd_addr_g1;
                        const_rd_addr_r <= rn_ap_c_addr_g1;
                    end
                    state        <= S_RN12;
                end
                S_RN12: state <= S_RN13;
                S_RN13: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= rn_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (ctr_g == rn_x_grp_last) begin
                        ctr_g     <= 7'd0;
                        ctr_k     <= 11'd0;
                        cur_stage <= STG_M1A;
                        state     <= S_MAC1;
                    end else begin
                        ctr_g <= ctr_g + 7'd1;
                        state  <= S_RN9;
                    end
                end

                S_M2_1: begin
                    tap_cnt         <= 2'd0;
                    m_rd_addr       <= m2_rd_addr_tap0;
                    w_rd_addr       <= m2_w_addr_tap0;
                    const_rd_addr_r <= m2_c_addr;    // prefetch bias — hold on const bus through TAP loop
                    state           <= S_M2_2;
                end
                S_M2_2: begin
                    m_rd_addr <= m2_rd_addr_tap1;
                    w_rd_addr <= m2_w_addr_tap1;
                    state     <= S_M2_3;
                end
                S_M2_3: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (tap_cnt == 2'd0);
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_H_ext  <= m2_pad ? 256'b0 : m_rd_data;
                    m_rd_addr    <= m2_rd_addr_tap2;
                    w_rd_addr    <= m2_w_addr_tap2;
                    tap_cnt      <= tap_p1[1:0];
                    if (tap_cnt == 2'd3) state <= S_M2_4;
                end
                S_M2_4: state <= S_M2_5;      // skip BIAS_PREF/WAIT (bias already fetched)
                S_M2_5: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= cl_out_vec;          // conv result (was in old BIAS_PREF)
                    cl_in_H_ext  <= const_rd_data;       // bias held on const bus since PREF prefetch
                    state        <= S_M2_6;
                end
                S_M2_6: state <= S_M2_7;
                S_M2_7: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m2_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (ctr_g == inner_grp_last) begin
                        ctr_g     <= 7'd0;
                        m3_wr_ptr <= 7'd0;
                        cur_stage <= STG_M3;
                        state     <= S_M3_1;
                    end else begin
                        // Fuse next-group PREF into this cycle: WRITE uses port A,
                        // read prefetch uses port B — dual-port safe.
                        ctr_g       <= ctr_g_p1;
                        tap_cnt         <= 2'd0;
                        m_rd_addr       <= m2_rd_addr_gnext;
                        w_rd_addr       <= m2_w_addr_gnext;
                        const_rd_addr_r <= m2_c_addr_gnext;
                        state           <= S_M2_2;
                    end
                end

                S_M3_1: begin
                    if (ctr_g <= inner_grp_last) begin
                        m_rd_addr <= PT_X_CONV + {8'b0, ctr_g};
                    end
                    ctr_g <= ctr_g + 7'd1;
                    if (ctr_g >= 7'd2) begin
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_U + {8'b0, m3_wr_ptr};
                        m_wr_data <= silu_out_w;
                        m3_wr_ptr <= m3_wr_ptr + 7'd1;
                        if (m3_wr_ptr == inner_grp_last) begin
                            ctr_g <= 7'd0;
                            ctr_k  <= 11'd0;
                            cur_stage <= STG_M4;
                            state     <= S_MAC1;
                        end
                    end
                end

                S_M5_1: state <= S_M5_2;
                S_M5_2: begin
                    m5_dt_word_reg <= m_rd_data;
                    ctr_l     <= 4'd0;
                    ctr_r       <= 5'd0;
                    state          <= S_M5_3;
                end
                S_M5_3: begin
                    w_rd_addr       <= m5_w_addr_r0;
                    const_rd_addr_r <= m5_c_addr;  // prefetch bias — hold on const bus through MAC
                    state           <= S_M5_4;
                end
                S_M5_4: begin
                    w_rd_addr <= m5_w_addr_r1;
                    state     <= S_M5_5;
                end
                S_M5_5: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (ctr_r == 5'd0);
                    cl_in_W1_vec <= w_rd_data;
                    cl_in_H_ext  <= m5_dt_broadcast;
                    w_rd_addr    <= m5_w_addr_r2;
                    ctr_l   <= ctr_l + 4'd1;
                    ctr_r     <= ctr_r_p1;
                    if (ctr_r == dt_rank_last) state <= S_M5_6;
                end
                S_M5_6: state <= S_M5_7;    // skip BIAS_PREF/WAIT (bias already fetched)
                S_M5_7: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= cl_out_vec;         // proj result (was in old BIAS_PREF)
                    cl_in_H_ext  <= const_rd_data;      // bias held on const bus since W_PREF prefetch
                    state        <= S_M5_8;
                end
                S_M5_8: begin
                    // Prefetch NEXT group's W[0] on port B (weight port free during PE drain)
                    if (ctr_g != inner_grp_last)
                        w_rd_addr <= m5_w_addr_gnext_r0;
                    state <= S_M5_9;
                end
                S_M5_9: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m5_wr_addr;
                    m_wr_data <= sp_out_w;
                    ctr_l <= 4'd0;
                    ctr_r   <= 5'd0;
                    if (ctr_g == inner_grp_last) begin
                        ctr_g <= 7'd0;
                        cur_stage <= STG_M6;
                        ctr_s     <= 3'd0;
                        ctr_load  <= 3'd0;
                        state     <= S_M6_1;
                    end else begin
                        // Fuse next-group W[1] issue + bias prefetch into WRITE cycle:
                        // WRITE uses port A for m_we, read prefetch uses port B — dual-port safe.
                        // Combined with BIAS_LATCH's W[0] issue, weight-port pipeline is primed
                        // so we can skip W_PREF + W_WAIT entirely, landing directly in MAC.
                        ctr_g       <= ctr_g_p1;
                        w_rd_addr       <= m5_w_addr_gnext_r1;
                        const_rd_addr_r <= m5_c_addr_gnext;
                        state           <= S_M5_5;
                    end
                end

                // -----------------------------------------------------------
                // M6 LOAD template — Mamba2 per-(l,s) reload
                //   Slots (issue→wait→latch, 3-cyc per slot):
                //     0 = B[s]     via m_rd_addr = m6_B_addr_s
                //     1 = C[s]     via m_rd_addr = m6_C_addr_s
                //     2 = A[s]     via w_rd_addr = m6_A_addr_s  (weight port)
                //     3 = dt + D   via m_rd_addr + const_rd_addr (once per group)
                //     4 = u        via m_rd_addr = m6_u_addr    (once per group)
                //   Full 5-slot load at group start (ctr_l==0 && ctr_s==0);
                //   3-slot per-(l,s) reload otherwise.
                //   H_RegFile read addr issued at slot 0 for h_old[l,s].
                // -----------------------------------------------------------
                S_M6_1: begin
                    case (ctr_load)
                        3'd0: begin
                            m_rd_addr      <= m6_B_addr_s;
                            m6_h_rd_addr_r <= m6_c;              // h_old[l,s]
                        end
                        3'd1: m_rd_addr <= m6_C_addr_s;
                        3'd2: w_rd_addr <= m6_A_addr_s;          // weight port for A
                        3'd3: begin
                            m_rd_addr       <= m6_delta_addr;
                            const_rd_addr_r <= m6_D_addr_const;
                        end
                        3'd4: m_rd_addr <= m6_u_addr;
                        default: ;
                    endcase
                    state <= S_M6_2;
                end
                S_M6_2: state <= S_M6_3;                          // WAIT (BRAM 2-cyc)
                S_M6_3: begin
                    case (ctr_load)
                        3'd0: m6_w0_reg     <= m_rd_data;         // B[s]
                        3'd1: m6_w1_reg     <= m_rd_data;         // C[s]
                        3'd2: m6_w2_reg     <= w_rd_data;         // A[s]
                        3'd3: begin
                            m6_delta_word   <= m_rd_data;
                            m6_D_word       <= const_rd_data;
                        end
                        3'd4: m6_u_word     <= m_rd_data;
                        default: ;
                    endcase
                    // Exit: full 5 (group start) or short 3 (per-l,s reload)
                    if (ctr_load == 3'd4) begin
                        state    <= S_M6_4;
                    end else if (ctr_load == 3'd2 &&
                                 !(ctr_l == 4'd0 && ctr_s == 3'd0)) begin
                        state    <= S_M6_4;
                    end else begin
                        ctr_load <= ctr_load + 3'd1;
                        state    <= S_M6_1;
                    end
                end
                S_M6_4: begin
                    // DAB MUL2: m1 = dt·A[s] → exp → dA;  m2 = dt·B[s] → dB
                    cl_op_mode   <= `MAMBA_PE_MUL2;
                    cl_in_W1_vec <= m6_dt_broadcast;
                    cl_in_H_ext  <= m6_w2_reg;                    // A[s] from LOAD
                    cl_in_W2_vec <= m6_dt_broadcast;
                    cl_in_X_vec  <= m6_w0_reg;                    // B[s] from LOAD
                    state        <= S_M6_5;
                end
                S_M6_5: state <= S_M6_6;
                S_M6_6: begin
                    m6_dA_reg      <= exp_out_w;
                    m6_dB_reg      <= cl_out_vec2;
                    m6_h_from_rf_r <= 1'b1;
                    state          <= S_M6_7;
                end
                // T1/T2 fusion via MUL2 + ADD:
                //   SSM_MUL2: op=MUL2 → cl_out_vec  = sat16((dA·h_old) >> FB) = T1
                //                       cl_out_vec2 = sat16((dB·u    ) >> FB) = T2
                //   ADD_ISSUE: op=ADD, W1<=T1, H<=T2 → cl_out_vec = sat16(T1+T2) = h_new
                //   SSM_LATCH: assert h_wr_en (fold old H_WRITE into same state).
                // Byte-exact match old T1_MUL/T2_MUL/H_ADD chain (2 intermediate sats
                // preserved) — NOT MAMBA_PE_SSM which single-sats and diverges.
                S_M6_7: begin
                    cl_op_mode   <= `MAMBA_PE_MUL2;
                    cl_in_W1_vec <= m6_dA_reg;       // m1 W1 = dA (H via h_from_rf=1)
                    cl_in_W2_vec <= m6_dB_reg;       // m2 W2 = dB
                    cl_in_X_vec  <= m6_u_broadcast;  // m2 X  = u
                    state        <= S_M6_8;
                end
                S_M6_8: state <= S_M6_9;
                S_M6_9: begin
                    cl_op_mode     <= `MAMBA_PE_ADD;
                    cl_in_W1_vec   <= cl_out_vec;     // T1 registered from MUL2
                    cl_in_H_ext    <= cl_out_vec2;    // T2 registered from MUL2
                    m6_h_from_rf_r <= 1'b0;
                    state          <= S_M6_10;
                end
                // Direct ADD_ISSUE → SSM_LATCH (no wait):
                // At ADD_ISSUE edge, PE mode = ADD, inputs = T1, T2.
                // During SSM_LATCH cycle, PE ADD combinationally computes
                // out_next = sat16(T1+T2) = h_new, registered at SSM_LATCH edge
                // → cl_out_vec = h_new. Y_MAC captures W1 <= cl_out_vec = h_new
                // and asserts wr_en → H_RegFile writes h_new (h_wr_data mux
                // picks cl_out_vec). Adding an ADD_WAIT state between would let
                // the default cl_op_mode <= IDLE clear PE mode before SSM_LATCH,
                // wiping cl_out_vec to 0 (IDLE out_next default).
                S_M6_10: begin
                    m6_h_wr_en_r      <= 1'b1;
                    m6_h_wr_addr_r    <= m6_c;                     // Widened 14-bit
                    m6_h_wr_from_pe_r <= 1'b1;
                    state             <= S_M6_11;
                end
                // Y_MAC per s: single MAC C[s]·h_new[s]. External accumulator
                // (m6_y_ch_reg) sums the 8 per-s partials at S_M6_13, since PE
                // acc would be clobbered by DAB MUL2 / SSM MUL2 / ADD ops in
                // the intervening LOAD → compute chain.
                S_M6_11: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= 1'b1;                         // fresh MAC per s
                    cl_in_W1_vec <= cl_out_vec;                   // h_new[l,s]
                    cl_in_H_ext  <= m6_w1_reg;                    // C[s]
                    state        <= S_M6_12;
                end
                S_M6_12: state <= S_M6_13;                        // PE MAC settle
                S_M6_13: begin
                    // Accumulate partial y[c] across s = 0..N_STATE_GRP-1
                    if (ctr_s == 3'd0) begin
                        m6_y_ch_reg <= {{8{m6_y_ch_sat[15]}}, m6_y_ch_sat};
                    end else begin
                        m6_y_ch_reg <= m6_y_ch_reg +
                                       {{8{m6_y_ch_sat[15]}}, m6_y_ch_sat};
                    end
                    if (ctr_s == `N_STATE_GRP_MAX) begin
                        // Done all s → continue to D·u + YSUM
                        state <= S_M6_14;
                    end else begin
                        // Next s: reload B/C/A for new s
                        ctr_s    <= ctr_s + 3'd1;
                        ctr_load <= 3'd0;
                        state    <= S_M6_1;
                    end
                end
                S_M6_14: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m6_D_broadcast;
                    cl_in_H_ext  <= m6_u_broadcast;
                    state        <= S_M6_15;
                end
                S_M6_15: state <= S_M6_16;
                // Fire PE ADD: sat16(y_ch_reg + cl_out_vec[15:0]).
                // cl_in_H_ext captures cl_out_vec[15:0]=D*u while still valid.
                // Lane-0 broadcast is enough since only lane 0 of PE output is read.
                S_M6_16: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= {16{m6_y_ch_reg_sat}};        // sat16 of 24-bit acc
                    cl_in_H_ext  <= {16{cl_out_vec[15:0]}};
                    state        <= S_M6_17;
                end
                S_M6_17: state <= S_M6_18;
                S_M6_18: begin
                    m6_ssm_grp_acc[ctr_l*16 +: 16] <= cl_out_vec[15:0];
                    if (ctr_l == `LANE_MAX) begin
                        state <= S_M6_19;
                    end else begin
                        // Next lane: reload B/C/A for new (l, s=0); dt/u still valid
                        ctr_l    <= ctr_l + 4'd1;
                        ctr_s    <= 3'd0;
                        ctr_load <= 3'd0;
                        state    <= S_M6_1;
                    end
                end
                S_M6_19: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m6_y_ssm_wr_addr;
                    m_wr_data <= m6_ssm_grp_acc;
                    if (ctr_g == inner_grp_last) begin
                        ctr_g <= 7'd0;
                        cur_stage <= STG_M7;
                        state     <= S_M7_1;
                    end else begin
                        // Next group: full 5-slot LOAD (dt/u/D refresh + B/C/A[s=0])
                        ctr_g    <= ctr_g + 7'd1;
                        ctr_l    <= 4'd0;
                        ctr_s    <= 3'd0;
                        ctr_load <= 3'd0;
                        state    <= S_M6_1;
                    end
                end

                S_M7_1: begin
                    m_rd_addr <= m7_z_addr;
                    state     <= S_M7_2;
                end
                S_M7_2: state <= S_M7_3;
                S_M7_3: begin
                    silu_z_reg <= silu_out_w;
                    m_rd_addr  <= m7_y_addr;
                    state      <= S_M7_4;
                end
                S_M7_4: state <= S_M7_5;
                S_M7_5: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= silu_z_reg;
                    if (ctr_g != inner_grp_last) begin
                        m_rd_addr <= m7_z_addr_p1;
                    end
                    state        <= S_M7_6;
                end
                S_M7_6: state <= S_M7_7;
                S_M7_7: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m7_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (ctr_g == inner_grp_last) begin
                        ctr_g <= 7'd0;
                        ctr_k  <= 11'd0;
                        cur_stage <= STG_M8;
                        state     <= S_MAC1;
                    end else begin
                        ctr_g <= ctr_g + 7'd1;
                        state     <= S_M7_3;
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
