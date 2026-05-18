`timescale 1ns / 1ps
`include "_parameter.v"

// ============================================================================
// ITMN_TB V9 - Full 5-block ITMN testbench.
//
// Golden file layout (from extract_itm_full.py output):
//   golden_all/
//     block_00_layer00/   T=1000, d_in=64,  d_inner=128, dt_rank=4
//     block_01_layer01/   T=1000, d_in=64,  d_inner=128, dt_rank=4
//     block_02_layer03/   T=500,  d_in=64,  d_inner=128, dt_rank=4
//     block_03_layer04/   T=500,  d_in=64,  d_inner=128, dt_rank=4
//     block_04_layer06/   T=250,  d_in=128, d_inner=256, dt_rank=8
//
// Flow: load ? DMA weights+input ? start ? wait phases (with cycle report) ?
//       compare all stages (with err+max_d table) ? MaxPool if needed ? repeat.
//
// Report: identical style to V8e (per stage: size | err | max_d | pass?)
//         plus final 5-block accumulated summary.
// ============================================================================
module ITMN_TB;

    // ======================================================================
    // DUT
    // ======================================================================
    reg clk, rst, start;
    wire done_phase1, done_inception, done_mamba, done_all;

    reg  dma_write_en;
    reg  [1:0]  dma_target;
    reg  [14:0] dma_addr;
    reg  [255:0] dma_wdata;
    wire dma_ready;

    // Per-block parameters driven by TB (set before each block's start pulse)
    reg  [9:0] BLK_T_REG;
    reg  [3:0] BLK_CH_IN_REG;
    reg  [3:0] BLK_CH_OUT_REG;
    reg  [3:0] BLK_CH_M_REG;
    reg  [3:0] BLK_DT_REG;

    initial begin clk = 0; forever #5 clk = ~clk; end

    // Instantiate the parameterized ITM_Top directly (V9 style)
    ITM_Top uut (
        .clk(clk), .rst(rst), .start(start),
        .done_phase1(done_phase1), .done_inception(done_inception),
        .done_mamba(done_mamba), .done_all(done_all),
        .T_MAX(BLK_T_REG), .CH_IN(BLK_CH_IN_REG), .CH_OUT(BLK_CH_OUT_REG),
        .CH_M(BLK_CH_M_REG), .DT_RANK(BLK_DT_REG),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_ready(dma_ready)
    );

    // ======================================================================
    // Address map (must match ITM_CONTROLLER.v)
    // ======================================================================
    localparam A_INPUT_BASE = 15'd0;
    localparam A_BOT_OUT    = 15'd4000;
    localparam A_CH1_OUT    = 15'd5000;
    localparam A_FINAL_OUT  = 15'd8000;
    localparam A_X_INNER    = 15'd12000;
    localparam A_Z_GATE     = 15'd20000;
    localparam A_MAMBA_OUT  = 15'd28128;
    localparam A_H_STATE    = 15'd28000;
    localparam B_P1_OUT     = 15'd0;
    localparam B_CH2_OUT    = 15'd4000;
    localparam B_CH3_OUT    = 15'd5000;
    localparam B_CH4_OUT    = 15'd6000;
    localparam B_FINAL_OUT  = 15'd8000;
    localparam B_X_CONV     = 15'd12000;
    localparam B_U_SAFE     = 15'd15000;
    localparam B_Y_SSM      = 15'd23000;
    // Const RAM layout - must match ITM_CONTROLLER.v.
    // Sized for block 4 (CH_OUT<=8, CH_M<=16) to avoid address overlaps.
    localparam C_P1_BIAS    = 15'd0;     // size CH_OUT
    localparam C_INC_SCALE  = 15'd8;     // size CH_OUT
    localparam C_INC_SHIFT  = 15'd16;    // size CH_OUT
    localparam C_M_DW_BIAS  = 15'd24;    // size CH_M
    localparam C_M_DT_BIAS  = 15'd40;    // size CH_M
    localparam C_NORM_W     = 15'd56;    // size CH_OUT - RMSNorm gamma

    localparam TOLERANCE  = 2;
    localparam MAX_CYCLES = 100_000_000;

    // ======================================================================
    // File data arrays - sized for largest block (block 4: d_in=128, d_inner=256)
    // ======================================================================
    reg signed [15:0] f_Xin      [0:127999];
    reg signed [15:0] f_Wp1      [0:16383];
    reg signed [15:0] f_Bconv    [0:127];
    reg signed [15:0] f_Wbot     [0:4095];
    reg signed [15:0] f_Wb1      [0:4095];
    reg signed [15:0] f_Wb2      [0:9215];
    reg signed [15:0] f_Wb3      [0:19455];
    reg signed [15:0] f_Wb4      [0:39935];
    reg signed [15:0] f_Inc_Scale[0:127];
    reg signed [15:0] f_Inc_Shift[0:127];
    reg signed [15:0] f_Wmx      [0:32767];
    reg signed [15:0] f_Wmz      [0:32767];
    reg signed [15:0] f_Wconv    [0:1023];
    reg signed [15:0] f_Bconv_dw [0:255];
    reg signed [15:0] f_Wxproj   [0:12287];
    reg signed [15:0] f_Wdt      [0:2047];
    reg signed [15:0] f_Bdt      [0:255];
    reg signed [15:0] f_Alog     [0:4095];
    reg signed [15:0] f_Dparam   [0:255];
    reg signed [15:0] f_Woutproj [0:32767];
    reg signed [15:0] f_Norm_W   [0:127];   // RMSNorm gamma (max CH_OUT=8, 128 values)

    // Goldens
    reg signed [15:0] goldfp_p1     [0:127999];
    reg signed [15:0] goldfp_bot    [0:31999];
    reg signed [15:0] goldfp_b1     [0:31999];
    reg signed [15:0] goldfp_b2     [0:31999];
    reg signed [15:0] goldfp_b3     [0:31999];
    reg signed [15:0] goldfp_b4     [0:31999];
    reg signed [15:0] goldfp_zgate  [0:255999];
    reg signed [15:0] goldfp_usilu  [0:255999];
    reg signed [15:0] goldfp_xproj  [0:47999];
    reg signed [15:0] goldfp_delta  [0:255999];
    reg signed [15:0] gold_h        [0:4095];
    reg signed [15:0] gold_y_gated  [0:255999];
    reg signed [15:0] gold_mout     [0:127999];
    reg signed [15:0] goldfp_final  [0:127999];

    // ======================================================================
    // Per-block params
    // ======================================================================
    integer BLK_T, BLK_CH_IN, BLK_CH_OUT, BLK_CH_M, BLK_D_INNER, BLK_DT_RANK;
    integer BLK_DIM, BLK_BR_GRPS;
    integer DW_P1, DW_BOT, DW_B1, DW_B2, DW_B3, DW_B4, DW_MX, DW_MZ, DW_DW, DW_XPROJ, DW_DTPROJ, DW_ALOG, DW_DPARAM, DW_OUTPROJ;
    reg [14:0] W_DTPROJ, W_ALOG, W_DPARAM, W_OUTPROJ;
    reg [14:0] dtproj_words_r;

    // ======================================================================
    // Loop/temp vars (all top-level for Verilog-2001)
    // ======================================================================
    integer i, t, c, k, c_grp, c_grp_m, c_grp_br, s, lane;
    integer t_idx, c_idx, diff;
    reg [255:0] rdata, pool_a, pool_b, pool_res;
    reg signed [15:0] r_got, r_exp, va, vb;
    reg [15:0] r_raw;
    integer sanity_x_cnt;
    integer cycle_p1, cycle_inc, cycle_mam, cycle_fin;
    integer total_cycles_all;
    integer blk;
    // Error counters
    integer err_p1,   max_p1;
    integer err_bot,  max_bot;
    integer err_b1,   max_b1;
    integer err_b2,   max_b2;
    integer err_b3,   max_b3;
    integer err_b4,   max_b4;
    integer err_zgate,max_zgate;
    integer err_usilu,max_usilu;
    integer err_xproj,max_xproj;
    integer err_delta,max_delta;
    integer err_h,    max_h;
    integer err_ygated,max_ygated;
    integer err_mout, max_mout;
    integer err_fin,  max_fin;
    // Accumulators
    integer tot_err_p1, tot_err_fin, tot_err_mout, tot_err_h;
    integer tot_err_yg, tot_err_dt, tot_err_xp, tot_err_us, tot_err_zg;
    integer worst_p1, worst_fin, worst_mout, worst_h;

    // ======================================================================
    // task dma_write
    // ======================================================================
    task dma_write;
        input [1:0]  tgt;
        input [14:0] addr;
        input [255:0] data;
        begin
            @(negedge clk);
            dma_write_en = 1; dma_target = tgt;
            dma_addr = addr; dma_wdata = data;
            @(posedge clk); #1;
            dma_write_en = 0;
        end
    endtask

    // ======================================================================
    // task set_block_params
    // ======================================================================
    task set_block_params;
        input integer bid;
        begin
            case (bid)
                0, 1: begin BLK_T=1000; BLK_CH_IN=4; BLK_CH_OUT=4; BLK_CH_M=8;  BLK_D_INNER=128; BLK_DT_RANK=4; end
                2, 3: begin BLK_T=500;  BLK_CH_IN=4; BLK_CH_OUT=4; BLK_CH_M=8;  BLK_D_INNER=128; BLK_DT_RANK=4; end
                4:    begin BLK_T=250;  BLK_CH_IN=4; BLK_CH_OUT=8; BLK_CH_M=16; BLK_D_INNER=256; BLK_DT_RANK=8; end
                default: begin BLK_T=1000; BLK_CH_IN=4; BLK_CH_OUT=4; BLK_CH_M=8; BLK_D_INNER=128; BLK_DT_RANK=4; end
            endcase
            // Drive DUT input regs. CH_M=16 wraps as 4'd0 (controller interprets ch_m_last=15).
            BLK_T_REG     = BLK_T[9:0];
            BLK_CH_IN_REG = BLK_CH_IN[3:0];
            BLK_CH_OUT_REG = BLK_CH_OUT[3:0];
            // Compute dynamic weight base addresses
            BLK_DIM = BLK_CH_OUT * 4;
            BLK_BR_GRPS = (BLK_CH_OUT >= 8) ? 2 : 1;
            DW_P1     = 0;
            DW_BOT    = DW_P1  + BLK_CH_OUT * BLK_CH_IN * 16;
            DW_B1     = DW_BOT + BLK_BR_GRPS * BLK_CH_OUT * 16;
            DW_B2     = DW_B1  + BLK_BR_GRPS * BLK_CH_OUT * 16;
            DW_B3     = DW_B2  + BLK_BR_GRPS * 9  * BLK_DIM;
            DW_B4     = DW_B3  + BLK_BR_GRPS * 19 * BLK_DIM;
            DW_MX     = DW_B4  + BLK_BR_GRPS * 39 * BLK_DIM;
            DW_MZ     = DW_MX  + BLK_CH_M * BLK_CH_OUT * 16;
            DW_DW     = DW_MZ  + BLK_CH_M * BLK_CH_OUT * 16;
            DW_XPROJ  = DW_DW  + BLK_CH_M * 4;
            DW_DTPROJ = DW_XPROJ + 3 * BLK_D_INNER;
            DW_ALOG   = DW_DTPROJ + BLK_CH_M * BLK_DT_RANK;
            DW_DPARAM = DW_ALOG + BLK_CH_M * 16;
            DW_OUTPROJ= DW_DPARAM + BLK_CH_M;
            BLK_CH_M_REG  = (BLK_CH_M == 16) ? 4'd0 : BLK_CH_M[3:0];
            BLK_DT_REG    = BLK_DT_RANK[3:0];
            // Derived weight addresses
            W_DTPROJ = (BLK_D_INNER == 128) ? 15'd2896 : 15'd3280;
            if      (BLK_D_INNER==128 && BLK_DT_RANK==4) dtproj_words_r = 15'd32;
            else if (BLK_D_INNER==256 && BLK_DT_RANK==8) dtproj_words_r = 15'd128;
            else                                           dtproj_words_r = 15'd64;
            W_ALOG    = W_DTPROJ + dtproj_words_r;
            W_DPARAM  = W_ALOG   + ((BLK_D_INNER==128) ? 15'd128 : 15'd256);
            W_OUTPROJ = W_DPARAM + ((BLK_D_INNER==128) ? 15'd8   : 15'd16);
        end
    endtask

    // ======================================================================
    // task reset_err
    // ======================================================================
    task reset_err;
        begin
            err_p1=0;max_p1=0; err_bot=0;max_bot=0; err_b1=0;max_b1=0;
            err_b2=0;max_b2=0; err_b3=0;max_b3=0; err_b4=0;max_b4=0;
            err_zgate=0;max_zgate=0; err_usilu=0;max_usilu=0;
            err_xproj=0;max_xproj=0; err_delta=0;max_delta=0;
            err_h=0;max_h=0; err_ygated=0;max_ygated=0;
            err_mout=0;max_mout=0; err_fin=0;max_fin=0;
        end
    endtask

    // ======================================================================
    // task load_weights_and_input
    // (Arrays must already be $readmemh'd before calling)
    // ======================================================================
    task load_weights_and_input;
        begin
            // -- Input X ? RAM A (data RAM target=0)
            for (t = 0; t < BLK_T; t = t + 1)
                for (c_grp = 0; c_grp < BLK_CH_IN; c_grp = c_grp + 1) begin
                    for (c = 0; c < 16; c = c + 1)
                        dma_wdata[c*16 +: 16] = f_Xin[(c_grp*16+c)*BLK_T + t];
                    dma_write(0, A_INPUT_BASE + t*BLK_CH_IN + c_grp, dma_wdata);
                end

            // -- P1 weight ? W RAM (target=2)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1)
                for (c = 0; c < BLK_CH_IN*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wp1[(c_grp*16+i)*(BLK_CH_IN*16)+c];
                    dma_write(2, DW_P1 + c_grp*(BLK_CH_IN*16) + c, dma_wdata);
                end

            // -- P1 bias + Inc BN scale/shift ? Const RAM (target=3)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1) begin
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Bconv[c_grp*16+i];
                dma_write(3, C_P1_BIAS + c_grp, dma_wdata);
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Inc_Scale[c_grp*16+i];
                dma_write(3, C_INC_SCALE + c_grp, dma_wdata);
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Inc_Shift[c_grp*16+i];
                dma_write(3, C_INC_SHIFT + c_grp, dma_wdata);
            end

            // -- Inception branch weights (Bot, B1: dim�d_out, dim=d_out/4)
            // Bot/B1 input = d_out channels. Output = dim channels = dim/16 output groups.
            // Each output group: 16 output channels, d_out input channels.
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wbot[(c_grp_br*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_BOT + c_grp_br*(BLK_CH_OUT*16) + c, dma_wdata);
                end
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wb1[(c_grp_br*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_B1 + c_grp_br*(BLK_CH_OUT*16) + c, dma_wdata);
                end
            // B2 (dim�dim, k=9), dim = BLK_CH_OUT*4
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (k = 0; k < 9; k = k + 1)
                    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
                        for (i = 0; i < 16; i = i + 1)
                            dma_wdata[i*16 +: 16] = f_Wb2[(c_grp_br*16+i)*(BLK_CH_OUT*4)*9 + c*9 + k];
                        dma_write(2, DW_B2 + c_grp_br*9*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
                    end
            // B3 (dim�dim, k=19), dim = BLK_CH_OUT*4
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (k = 0; k < 19; k = k + 1)
                    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
                        for (i = 0; i < 16; i = i + 1)
                            dma_wdata[i*16 +: 16] = f_Wb3[(c_grp_br*16+i)*(BLK_CH_OUT*4)*19 + c*19 + k];
                        dma_write(2, DW_B3 + c_grp_br*19*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
                    end
            // B4 (dim�dim, k=39), dim = BLK_CH_OUT*4
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (k = 0; k < 39; k = k + 1)
                    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
                        for (i = 0; i < 16; i = i + 1)
                            dma_wdata[i*16 +: 16] = f_Wb4[(c_grp_br*16+i)*(BLK_CH_OUT*4)*39 + c*39 + k];
                        dma_write(2, DW_B4 + c_grp_br*39*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
                    end

            // -- InProj X (d_inner � d_in)
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wmx[(c_grp_m*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_MX + c_grp_m*(BLK_CH_OUT*16) + c, dma_wdata);
                end
            // -- InProj Z (same shape)
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wmz[(c_grp_m*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_MZ + c_grp_m*(BLK_CH_OUT*16) + c, dma_wdata);
                end

            // -- DW Conv weight (d_inner � 4) + bias
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1) begin
                for (k = 0; k < 4; k = k + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wconv[(c_grp_m*16+i)*4+k];
                    dma_write(2, DW_DW + c_grp_m*4 + k, dma_wdata);
                end
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Bconv_dw[c_grp_m*16+i];
                dma_write(3, C_M_DW_BIAS + c_grp_m, dma_wdata);
            end

            // -- X_proj (3 groups � d_inner)
            for (c_grp = 0; c_grp < 3; c_grp = c_grp + 1)
                for (c = 0; c < BLK_D_INNER; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wxproj[(c_grp*16+i)*BLK_D_INNER+c];
                    dma_write(2, DW_XPROJ + c_grp*BLK_D_INNER + c, dma_wdata);
                end

            // -- DtProj (d_inner � dt_rank) + bias
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1) begin
                for (k = 0; k < BLK_DT_RANK; k = k + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wdt[(c_grp_m*16+i)*BLK_DT_RANK+k];
                    dma_write(2, DW_DTPROJ + c_grp_m*BLK_DT_RANK + k, dma_wdata);
                end
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Bdt[c_grp_m*16+i];
                dma_write(3, C_M_DT_BIAS + c_grp_m, dma_wdata);
            end

            // -- A_signed (d_inner � 16 states)
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1)
                for (k = 0; k < 16; k = k + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Alog[(c_grp_m*16+i)*16+k];
                    dma_write(2, DW_ALOG + c_grp_m*16 + k, dma_wdata);
                end

            // -- D param (d_inner)
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1) begin
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Dparam[c_grp_m*16+i];
                dma_write(2, DW_DPARAM + c_grp_m, dma_wdata);
            end

            // -- OutProj (d_in � d_inner)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1)
                for (c = 0; c < BLK_D_INNER; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Woutproj[(c_grp*16+i)*BLK_D_INNER+c];
                    dma_write(2, DW_OUTPROJ + c_grp*BLK_D_INNER + c, dma_wdata);
                end

            // -- RMSNorm gamma weights -> Const RAM (target=3)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1) begin
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Norm_W[c_grp*16+i];
                dma_write(3, C_NORM_W + c_grp, dma_wdata);
            end
        end
    endtask

    // ======================================================================
    // task sanity_check - X-check goldens + print sample values
    // ======================================================================
    task sanity_check;
        input integer T;
        input integer ch_m;
        input integer ch_out;
        begin
            sanity_x_cnt = 0;
            for (i = 0; i < ch_out*16*T;  i = i + 1) begin
                if (^goldfp_p1[i]    === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
                if (^goldfp_final[i] === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
                if (^gold_mout[i]    === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
            end
            for (i = 0; i < ch_m*16*T; i = i + 1) begin
                if (^goldfp_zgate[i]  === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
                if (^goldfp_usilu[i]  === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
                if (^goldfp_delta[i]  === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
                if (^gold_y_gated[i]  === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
            end
            for (i = 0; i < 48*T;      i = i + 1)
                if (^goldfp_xproj[i] === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
            for (i = 0; i < ch_m*16*16;i = i + 1)
                if (^gold_h[i]       === 1'bx) sanity_x_cnt = sanity_x_cnt + 1;
            if (sanity_x_cnt > 0)
                $display("   [WARN] %0d X-values in goldens (bad file?)", sanity_x_cnt);
            else
                $display("   [OK]   0 X-values in all golden arrays.");
            $display("   Sample P1[0..1]     : %04h %04h", goldfp_p1[0],   goldfp_p1[1]);
            $display("   Sample ZGate[0..1]  : %04h %04h", goldfp_zgate[0],goldfp_zgate[1]);
            $display("   Sample USilu[0..1]  : %04h %04h", goldfp_usilu[0],goldfp_usilu[1]);
            $display("   Sample XProj[0..1]  : %04h %04h", goldfp_xproj[0],goldfp_xproj[1]);
            $display("   Sample Delta[0..1]  : %04h %04h", goldfp_delta[0], goldfp_delta[1]);
            $display("   Sample H[0..1]      : %04h %04h", gold_h[0],       gold_h[1]);
            $display("   Sample YGated[0..1] : %04h %04h", gold_y_gated[0], gold_y_gated[1]);
            $display("   Sample Mout[0..1]   : %04h %04h", gold_mout[0],    gold_mout[1]);
            $display("   Sample Final[0..1]  : %04h %04h", goldfp_final[0], goldfp_final[1]);
        end
    endtask

    // ======================================================================
    // task run_one_block - wait for inner done signals, per-phase timing
    // ======================================================================
    task run_one_block;
        begin
            // Pulse start (this triggers S_IDLE?S_P1_MAC and clears all done flags)
            @(negedge clk); start = 1;
            @(posedge clk); #1;
            @(negedge clk); start = 0;
            // After start pulse, done flags should be 0 within 1-2 cycles
            // Wait briefly for FSM to leave IDLE
            repeat(3) @(posedge clk);

            cycle_p1 = 0;
            while (!uut.done_phase1 && cycle_p1 < MAX_CYCLES) begin
                @(posedge clk); cycle_p1 = cycle_p1 + 1;
            end
            if (cycle_p1 >= MAX_CYCLES) begin
                $display("[ERROR] P1 timeout block %0d!", blk); $stop;
            end
            $display("   [BLK %0d] Phase1 done  : %7d cyc", blk, cycle_p1);

            cycle_inc = 0;
            while (!uut.done_inception && cycle_inc < MAX_CYCLES) begin
                @(posedge clk); cycle_inc = cycle_inc + 1;
                if (cycle_inc % 1000000 == 0)
                    $display("   [BLK %0d] Inc running... %0dM cyc  state=%0d br=%0d t=%0d",
                             blk, cycle_inc/1000000, uut.state,
                             uut.branch_id, uut.t_cnt);
            end
            $display("   [BLK %0d] Inception done: %7d cyc", blk, cycle_inc);

            cycle_mam = 0;
            while (!uut.done_mamba && cycle_mam < MAX_CYCLES) begin
                @(posedge clk); cycle_mam = cycle_mam + 1;
                if (cycle_mam % 500000 == 0)
                    $display("   [BLK %0d] Mamba running... %0dM cyc  state=%0d t=%0d g=%0d s=%0d",
                             blk, cycle_mam/1000000, uut.state,
                             uut.t_cnt, uut.c_grp_m, uut.s_idx);
            end
            $display("   [BLK %0d] Mamba done    : %7d cyc", blk, cycle_mam);

            cycle_fin = 0;
            while (!uut.done_all && cycle_fin < MAX_CYCLES) begin
                @(posedge clk); cycle_fin = cycle_fin + 1;
            end
            $display("   [BLK %0d] Final done    : %7d cyc", blk, cycle_fin);
            $display("   [BLK %0d] Total         : %7d cyc", blk,
                     cycle_p1+cycle_inc+cycle_mam+cycle_fin);
            repeat(10) @(posedge clk);
        end
    endtask

    // ======================================================================
    // task compare_all_stages - read HW RAM and compare vs goldens
    // ======================================================================
    task compare_all_stages;
        input integer T;
        input integer ch_out;  // d_out/16 (CH_OUT)
        input integer ch_m;   // d_inner/16 (or 16 for block4)
        integer dim_cmp, br_grps;
        begin
            dim_cmp = ch_out * 4;   // dim = d_out/4
            br_grps = (ch_out >= 8) ? 2 : 1;  // words per branch per timestep
            // -- P1 output (RAM B, B_P1_OUT + t*ch_out + (c>>4))
            for (c_idx = 0; c_idx < ch_out*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_P1_OUT + t_idx*ch_out + (c_idx>>4)];
                    r_raw = rdata[(c_idx & 4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_p1[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_p1) max_p1 = diff;
                    if (diff > TOLERANCE) err_p1 = err_p1 + 1;
                end

            // -- Bot (RAM A, A_BOT_OUT + t*br_grps + sub_grp)
            for (c_idx = 0; c_idx < dim_cmp; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_a.ram[A_BOT_OUT + t_idx*br_grps + (c_idx>>4)];
                    r_raw = rdata[(c_idx & 4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_bot[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_bot) max_bot = diff;
                    if (diff > TOLERANCE) err_bot = err_bot + 1;
                end

            // -- B1 (RAM A, A_CH1_OUT + t*br_grps + sub_grp)
            for (c_idx = 0; c_idx < dim_cmp; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_a.ram[A_CH1_OUT + t_idx*br_grps + (c_idx>>4)];
                    r_raw = rdata[(c_idx & 4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_b1[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_b1) max_b1 = diff;
                    if (diff > TOLERANCE) err_b1 = err_b1 + 1;
                end

            // -- B2 (RAM B, B_CH2_OUT + t*br_grps + sub_grp)
            for (c_idx = 0; c_idx < dim_cmp; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_CH2_OUT + t_idx*br_grps + (c_idx>>4)];
                    r_raw = rdata[(c_idx & 4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_b2[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_b2) max_b2 = diff;
                    if (diff > TOLERANCE) err_b2 = err_b2 + 1;
                end

            // -- B3 (RAM B, B_CH3_OUT + t*br_grps + sub_grp)
            for (c_idx = 0; c_idx < dim_cmp; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_CH3_OUT + t_idx*br_grps + (c_idx>>4)];
                    r_raw = rdata[(c_idx & 4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_b3[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_b3) max_b3 = diff;
                    if (diff > TOLERANCE) begin
                        err_b3 = err_b3 + 1;
                        if (err_b3 <= 30)
                            $display("[B3 ERR] c=%0d t=%0d | got=%0d (0x%04h) exp=%0d (0x%04h) diff=%0d addr=%0d",
                                     c_idx, t_idx, r_got, r_got & 16'hFFFF, r_exp, r_exp & 16'hFFFF, diff,
                                     B_CH3_OUT + t_idx*br_grps + (c_idx>>4));
                    end
                end

            // -- B4 (RAM B, B_CH4_OUT + t*br_grps + sub_grp)
            for (c_idx = 0; c_idx < dim_cmp; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_CH4_OUT + t_idx*br_grps + (c_idx>>4)];
                    r_raw = rdata[(c_idx & 4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_b4[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_b4) max_b4 = diff;
                    if (diff > TOLERANCE) err_b4 = err_b4 + 1;
                end

            // -- Z_Gate (RAM A, A_Z_GATE + t*ch_m + c>>4, lane=c&15)
            for (c_idx = 0; c_idx < ch_m*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_a.ram[A_Z_GATE + t_idx*ch_m + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_zgate[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_zgate) max_zgate = diff;
                    if (diff > TOLERANCE) err_zgate = err_zgate + 1;
                end

            // -- U_Safe (RAM B, B_U_SAFE + t*ch_m + c>>4)
            for (c_idx = 0; c_idx < ch_m*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_U_SAFE + t_idx*ch_m + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_usilu[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_usilu) max_usilu = diff;
                    if (diff > TOLERANCE) begin
                        err_usilu = err_usilu + 1;
                        if (err_usilu <= 20)
                            $display("[USAFE ERR] c=%0d t=%0d | got=%0d exp=%0d diff=%0d",
                                     c_idx, t_idx, r_got, r_exp, diff);
                    end
                end

            // -- X_Proj (RAM B, B_X_CONV + t*3 + c>>4, for 48 rows padded)
            for (c_idx = 0; c_idx < 48; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_X_CONV + t_idx*3 + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_xproj[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_xproj) max_xproj = diff;
                    if (diff > TOLERANCE) err_xproj = err_xproj + 1;
                end

            // -- Delta (RAM A, A_X_INNER + t*ch_m + c>>4 - final state after M5)
            for (c_idx = 0; c_idx < ch_m*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_a.ram[A_X_INNER + t_idx*ch_m + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_delta[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_delta) max_delta = diff;
                    if (diff > TOLERANCE) err_delta = err_delta + 1;
                end

            // -- H state (RAM A, A_H_STATE + s*ch_m + c>>4, golden: h[c*16+s])
            for (c_idx = 0; c_idx < ch_m*16; c_idx = c_idx + 1)
                for (s = 0; s < 16; s = s + 1) begin
                    rdata = uut.mem_sys.ram_a.ram[A_H_STATE + s*ch_m + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = gold_h[c_idx*16 + s];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_h) max_h = diff;
                    if (diff > TOLERANCE) err_h = err_h + 1;
                end

            // -- Y_Gated (RAM B, B_Y_SSM + t*ch_m + c>>4, after M7 overwrites)
            for (c_idx = 0; c_idx < ch_m*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_b.ram[B_Y_SSM + t_idx*ch_m + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = gold_y_gated[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_ygated) max_ygated = diff;
                    if (diff > TOLERANCE) err_ygated = err_ygated + 1;
                end

            // -- Mamba Out / OutProj (RAM A, A_MAMBA_OUT + t*ch_out + c>>4)
            for (c_idx = 0; c_idx < ch_out*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    rdata = uut.mem_sys.ram_a.ram[A_MAMBA_OUT + t_idx*ch_out + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = gold_mout[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_mout) max_mout = diff;
                    if (diff > TOLERANCE) err_mout = err_mout + 1;
                end

            // -- Final (RAM B for c_grp=0, RAM A for c_grp 1..3)
            for (c_idx = 0; c_idx < ch_out*16; c_idx = c_idx + 1)
                for (t_idx = 0; t_idx < T; t_idx = t_idx + 1) begin
                    if ((c_idx>>4) == 0)
                        rdata = uut.mem_sys.ram_b.ram[B_FINAL_OUT + t_idx*ch_out + 0];
                    else
                        rdata = uut.mem_sys.ram_a.ram[A_FINAL_OUT + t_idx*ch_out + (c_idx>>4)];
                    r_raw = rdata[(c_idx&4'hF)*16 +: 16];
                    r_got = (^r_raw === 1'bx) ? 16'sd0 : $signed(r_raw);
                    r_exp = goldfp_final[c_idx*T + t_idx];
                    diff = (r_got>r_exp) ? (r_got-r_exp) : (r_exp-r_got);
                    if (diff > max_fin) max_fin = diff;
                    if (diff > TOLERANCE) err_fin = err_fin + 1;
                end
        end
    endtask

    // ======================================================================
    // task print_block_report
    // ======================================================================
    task print_block_report;
        input integer bid;
        input integer T;
        input integer ch_out;
        input integer ch_m;
        begin
            $display("\n================================================================");
            $display("  BLOCK %0d REPORT   T=%0d  d_in=%0d  d_inner=%0d",
                     bid, T, ch_out*16, ch_m*16);
            $display("  FRAC_BITS=%0d  TOLERANCE=%0d", `FRAC_BITS, TOLERANCE);
            $display("================================================================");
            $display("  Timing:");
            $display("    Phase 1            : %8d cyc", cycle_p1);
            $display("    Inception          : %8d cyc", cycle_inc);
            $display("    Mamba              : %8d cyc", cycle_mam);
            $display("    Final BN+ReLU      : %8d cyc", cycle_fin);
            $display("    Block total        : %8d cyc", cycle_p1+cycle_inc+cycle_mam+cycle_fin);
            $display("  ----------------------------------------------------------------");
            $display("  Stage              | size     | err     | max_d  | result");
            $display("  -------------------+----------+---------+--------+-------");
            $display("  P1 Output          | %7d  |  %6d |  %5d | %s",
                     ch_out*16*T, err_p1,    max_p1,    (err_p1==0)   ?"PASS":"FAIL");
            $display("  Inc Bot            | %7d  |  %6d |  %5d | %s",
                     ch_out*4*T,  err_bot,   max_bot,   (err_bot==0)  ?"PASS":"FAIL");
            $display("  Inc B1             | %7d  |  %6d |  %5d | %s",
                     ch_out*4*T,  err_b1,    max_b1,    (err_b1==0)   ?"PASS":"FAIL");
            $display("  Inc B2             | %7d  |  %6d |  %5d | %s",
                     ch_out*4*T,  err_b2,    max_b2,    (err_b2==0)   ?"PASS":"FAIL");
            $display("  Inc B3             | %7d  |  %6d |  %5d | %s",
                     ch_out*4*T,  err_b3,    max_b3,    (err_b3==0)   ?"PASS":"FAIL");
            $display("  Inc B4             | %7d  |  %6d |  %5d | %s",
                     ch_out*4*T,  err_b4,    max_b4,    (err_b4==0)   ?"PASS":"FAIL");
            $display("  -------------------+----------+---------+--------+-------");
            $display("  Mam Z_Gate  (M1b)  | %7d  |  %6d |  %5d | %s",
                     ch_m*16*T,  err_zgate, max_zgate, (err_zgate==0)?"PASS":"FAIL");
            $display("  Mam U_Safe  (M3cp) | %7d  |  %6d |  %5d | %s",
                     ch_m*16*T,  err_usilu, max_usilu, (err_usilu==0)?"PASS":"FAIL");
            $display("  Mam X_Proj  (M4)   | %7d  |  %6d |  %5d | %s",
                     48*T,       err_xproj, max_xproj, (err_xproj==0)?"PASS":"FAIL");
            $display("  Mam Delta   (M5)   | %7d  |  %6d |  %5d | %s",
                     ch_m*16*T,  err_delta, max_delta, (err_delta==0)?"PASS":"FAIL");
            $display("  Mam H_State (M6a)  | %7d  |  %6d |  %5d | %s",
                     ch_m*16*16, err_h,     max_h,     (err_h==0)    ?"PASS":"FAIL");
            $display("  Mam Y_Gated (M7)   | %7d  |  %6d |  %5d | %s",
                     ch_m*16*T,  err_ygated,max_ygated,(err_ygated==0)?"PASS":"FAIL");
            $display("  Mam OutProj (M8)   | %7d  |  %6d |  %5d | %s",
                     ch_out*16*T, err_mout,  max_mout,  (err_mout==0) ?"PASS":"FAIL");
            $display("  -------------------+----------+---------+--------+-------");
            $display("  Final Full Output  | %7d  |  %6d |  %5d | %s",
                     ch_out*16*T, err_fin,   max_fin,   (err_fin==0)  ?"PASS":"FAIL");
            $display("================================================================");
        end
    endtask

    // ======================================================================
    // task accum_err - accumulate into totals
    // ======================================================================
    task accum_err;
        begin
            total_cycles_all = total_cycles_all + cycle_p1+cycle_inc+cycle_mam+cycle_fin;
            tot_err_p1  = tot_err_p1  + err_p1;
            tot_err_fin = tot_err_fin + err_fin;
            tot_err_mout= tot_err_mout+ err_mout;
            tot_err_h   = tot_err_h   + err_h;
            tot_err_yg  = tot_err_yg  + err_ygated;
            tot_err_dt  = tot_err_dt  + err_delta;
            tot_err_xp  = tot_err_xp  + err_xproj;
            tot_err_us  = tot_err_us  + err_usilu;
            tot_err_zg  = tot_err_zg  + err_zgate;
            if (max_p1   > worst_p1)   worst_p1   = max_p1;
            if (max_fin  > worst_fin)  worst_fin  = max_fin;
            if (max_mout > worst_mout) worst_mout = max_mout;
            if (max_h    > worst_h)    worst_h    = max_h;
        end
    endtask

    // ======================================================================
    // task maxpool_tb - TB-side MaxPool (read RAM ? max(t,t+1) ? write input)
    // ======================================================================
    task maxpool_tb;
        input integer T_in;
        input integer ch;
        begin
            $display("[TB] MaxPool: T_in=%0d ? T_out=%0d ch_grps=%0d", T_in, T_in/2, ch);
            for (t = 0; t < T_in; t = t + 2) begin
                for (c_grp = 0; c_grp < ch; c_grp = c_grp + 1) begin
                    if (c_grp == 0) begin
                        pool_a = uut.mem_sys.ram_b.ram[B_FINAL_OUT + t*ch     + 0];
                        pool_b = uut.mem_sys.ram_b.ram[B_FINAL_OUT + (t+1)*ch + 0];
                    end else begin
                        pool_a = uut.mem_sys.ram_a.ram[A_FINAL_OUT + t*ch     + c_grp];
                        pool_b = uut.mem_sys.ram_a.ram[A_FINAL_OUT + (t+1)*ch + c_grp];
                    end
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        va = $signed(pool_a[lane*16 +: 16]);
                        vb = $signed(pool_b[lane*16 +: 16]);
                        pool_res[lane*16 +: 16] = (va >= vb) ? va : vb;
                    end
                    dma_write(0, A_INPUT_BASE + (t/2)*ch + c_grp, pool_res);
                end
            end
            $display("[TB] MaxPool done - %0d words written to A_INPUT_BASE.", (T_in/2)*ch);
        end
    endtask

    // ======================================================================
    // task copy_final_to_input - copy final output to input for block chaining
    // (same-T transition, no MaxPool; reads B_FINAL_OUT/A_FINAL_OUT -> A_INPUT_BASE)
    // ======================================================================
    task copy_final_to_input;
        input integer T_val;
        input integer ch;
        begin
            $display("[TB] copy_final_to_input: T=%0d ch_grps=%0d", T_val, ch);
            for (t = 0; t < T_val; t = t + 1) begin
                for (c_grp = 0; c_grp < ch; c_grp = c_grp + 1) begin
                    if (c_grp == 0)
                        rdata = uut.mem_sys.ram_b.ram[B_FINAL_OUT + t*ch + 0];
                    else
                        rdata = uut.mem_sys.ram_a.ram[A_FINAL_OUT + t*ch + c_grp];
                    dma_write(0, A_INPUT_BASE + t*ch + c_grp, rdata);
                end
            end
            $display("[TB] copy_final_to_input done - %0d words written.", T_val*ch);
        end
    endtask

    // ======================================================================
    // task load_weights_only - same as load_weights_and_input without X DMA
    // (use when input is already in RAM A from previous block or MaxPool)
    // ======================================================================
    task load_weights_only;
        begin
            // -- P1 weight -> W RAM (target=2)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1)
                for (c = 0; c < BLK_CH_IN*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wp1[(c_grp*16+i)*(BLK_CH_IN*16)+c];
                    dma_write(2, DW_P1 + c_grp*(BLK_CH_IN*16) + c, dma_wdata);
                end

            // -- P1 bias + Inc BN scale/shift -> Const RAM (target=3)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1) begin
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Bconv[c_grp*16+i];
                dma_write(3, C_P1_BIAS + c_grp, dma_wdata);
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Inc_Scale[c_grp*16+i];
                dma_write(3, C_INC_SCALE + c_grp, dma_wdata);
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Inc_Shift[c_grp*16+i];
                dma_write(3, C_INC_SHIFT + c_grp, dma_wdata);
            end

            // -- Inception branch weights
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wbot[(c_grp_br*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_BOT + c_grp_br*(BLK_CH_OUT*16) + c, dma_wdata);
                end
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wb1[(c_grp_br*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_B1 + c_grp_br*(BLK_CH_OUT*16) + c, dma_wdata);
                end
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (k = 0; k < 9; k = k + 1)
                    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
                        for (i = 0; i < 16; i = i + 1)
                            dma_wdata[i*16 +: 16] = f_Wb2[(c_grp_br*16+i)*(BLK_CH_OUT*4)*9 + c*9 + k];
                        dma_write(2, DW_B2 + c_grp_br*9*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
                    end
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (k = 0; k < 19; k = k + 1)
                    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
                        for (i = 0; i < 16; i = i + 1)
                            dma_wdata[i*16 +: 16] = f_Wb3[(c_grp_br*16+i)*(BLK_CH_OUT*4)*19 + c*19 + k];
                        dma_write(2, DW_B3 + c_grp_br*19*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
                    end
            for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
                for (k = 0; k < 39; k = k + 1)
                    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
                        for (i = 0; i < 16; i = i + 1)
                            dma_wdata[i*16 +: 16] = f_Wb4[(c_grp_br*16+i)*(BLK_CH_OUT*4)*39 + c*39 + k];
                        dma_write(2, DW_B4 + c_grp_br*39*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
                    end

            // -- InProj X
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wmx[(c_grp_m*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_MX + c_grp_m*(BLK_CH_OUT*16) + c, dma_wdata);
                end
            // -- InProj Z
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1)
                for (c = 0; c < BLK_CH_OUT*16; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wmz[(c_grp_m*16+i)*(BLK_CH_OUT*16)+c];
                    dma_write(2, DW_MZ + c_grp_m*(BLK_CH_OUT*16) + c, dma_wdata);
                end

            // -- DW Conv weight + bias
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1) begin
                for (k = 0; k < 4; k = k + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wconv[(c_grp_m*16+i)*4+k];
                    dma_write(2, DW_DW + c_grp_m*4 + k, dma_wdata);
                end
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Bconv_dw[c_grp_m*16+i];
                dma_write(3, C_M_DW_BIAS + c_grp_m, dma_wdata);
            end

            // -- X_proj
            for (c_grp = 0; c_grp < 3; c_grp = c_grp + 1)
                for (c = 0; c < BLK_D_INNER; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wxproj[(c_grp*16+i)*BLK_D_INNER+c];
                    dma_write(2, DW_XPROJ + c_grp*BLK_D_INNER + c, dma_wdata);
                end

            // -- DtProj weight + bias
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1) begin
                for (k = 0; k < BLK_DT_RANK; k = k + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Wdt[(c_grp_m*16+i)*BLK_DT_RANK+k];
                    dma_write(2, DW_DTPROJ + c_grp_m*BLK_DT_RANK + k, dma_wdata);
                end
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Bdt[c_grp_m*16+i];
                dma_write(3, C_M_DT_BIAS + c_grp_m, dma_wdata);
            end

            // -- A_signed
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1)
                for (k = 0; k < 16; k = k + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Alog[(c_grp_m*16+i)*16+k];
                    dma_write(2, DW_ALOG + c_grp_m*16 + k, dma_wdata);
                end

            // -- D param
            for (c_grp_m = 0; c_grp_m < BLK_CH_M; c_grp_m = c_grp_m + 1) begin
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Dparam[c_grp_m*16+i];
                dma_write(2, DW_DPARAM + c_grp_m, dma_wdata);
            end

            // -- OutProj
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1)
                for (c = 0; c < BLK_D_INNER; c = c + 1) begin
                    for (i = 0; i < 16; i = i + 1)
                        dma_wdata[i*16 +: 16] = f_Woutproj[(c_grp*16+i)*BLK_D_INNER+c];
                    dma_write(2, DW_OUTPROJ + c_grp*BLK_D_INNER + c, dma_wdata);
                end

            // -- RMSNorm gamma weights -> Const RAM (target=3)
            for (c_grp = 0; c_grp < BLK_CH_OUT; c_grp = c_grp + 1) begin
                for (i = 0; i < 16; i = i + 1) dma_wdata[i*16 +: 16] = f_Norm_W[c_grp*16+i];
                dma_write(3, C_NORM_W + c_grp, dma_wdata);
            end
        end
    endtask

    // ======================================================================
    // MAIN SIMULATION
    // ======================================================================
    initial begin
        $display("================================================================");
        $display("  ITMN_TB V9 - Full 5-block pipeline");
        $display("  FRAC_BITS=%0d  TOLERANCE=%0d  MAX_CYCLES=%0d",
                 `FRAC_BITS, TOLERANCE, MAX_CYCLES);
        $display("================================================================");

        dma_write_en=0; dma_target=0; dma_addr=0; dma_wdata=0; start=0;
        total_cycles_all=0;
        tot_err_p1=0; tot_err_fin=0; tot_err_mout=0; tot_err_h=0;
        tot_err_yg=0; tot_err_dt=0; tot_err_xp=0; tot_err_us=0; tot_err_zg=0;
        worst_p1=0; worst_fin=0; worst_mout=0; worst_h=0;

        rst=1; repeat(5) @(posedge clk); #1;
        rst=0; repeat(3) @(posedge clk); #1;

        // ??????????????????????????????????????????????????????????????????
        // BLOCK 0  block_00_layer00   T=1000  d_in=64  d_inner=128
        // ??????????????????????????????????????????????????????????????????
        blk = 0; set_block_params(blk);
        $display("\n>> [BLOCK 0] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
                 BLK_T, BLK_CH_OUT*16, BLK_D_INNER, BLK_DT_RANK);

        $readmemh("golden_all/block_00_layer00/P1_Input_X.txt",         f_Xin);
        $readmemh("golden_all/block_00_layer00/P1_Weight_Fused.txt",    f_Wp1);
        $readmemh("golden_all/block_00_layer00/P1_Bias_Fused.txt",      f_Bconv);
        $readmemh("golden_all/block_00_layer00/W_Bot.txt",              f_Wbot);
        $readmemh("golden_all/block_00_layer00/W_B1.txt",               f_Wb1);
        $readmemh("golden_all/block_00_layer00/W_B2.txt",               f_Wb2);
        $readmemh("golden_all/block_00_layer00/W_B3.txt",               f_Wb3);
        $readmemh("golden_all/block_00_layer00/W_B4.txt",               f_Wb4);
        $readmemh("golden_all/block_00_layer00/Inc_BN_Scale.txt",       f_Inc_Scale);
        $readmemh("golden_all/block_00_layer00/Inc_BN_Shift.txt",       f_Inc_Shift);
        $readmemh("golden_all/block_00_layer00/Mam_W_InProj_X.txt",     f_Wmx);
        $readmemh("golden_all/block_00_layer00/Mam_W_InProj_Z.txt",     f_Wmz);
        $readmemh("golden_all/block_00_layer00/Mam_W_Conv.txt",         f_Wconv);
        $readmemh("golden_all/block_00_layer00/Mam_B_Conv.txt",         f_Bconv_dw);
        $readmemh("golden_all/block_00_layer00/Mam_W_XProj.txt",        f_Wxproj);
        $readmemh("golden_all/block_00_layer00/Mam_W_DtProj.txt",       f_Wdt);
        $readmemh("golden_all/block_00_layer00/Mam_B_DtProj.txt",       f_Bdt);
        $readmemh("golden_all/block_00_layer00/Mam_A_signed.txt",       f_Alog);
        $readmemh("golden_all/block_00_layer00/Mam_D_param.txt",        f_Dparam);
        $readmemh("golden_all/block_00_layer00/Mam_W_OutProj.txt",      f_Woutproj);
        $readmemh("golden_all/block_00_layer00/P1_Output_Golden_FP.txt",goldfp_p1);
        $readmemh("golden_all/block_00_layer00/Out_Bot_FP.txt",         goldfp_bot);
        $readmemh("golden_all/block_00_layer00/Out_B1_FP.txt",          goldfp_b1);
        $readmemh("golden_all/block_00_layer00/Out_B2_FP.txt",          goldfp_b2);
        $readmemh("golden_all/block_00_layer00/Out_B3_FP.txt",          goldfp_b3);
        $readmemh("golden_all/block_00_layer00/Out_B4_FP.txt",          goldfp_b4);
        $readmemh("golden_all/block_00_layer00/Mam_Z_Gate_FP.txt",      goldfp_zgate);
        $readmemh("golden_all/block_00_layer00/Mam_U_Silu_FP.txt",      goldfp_usilu);
        $readmemh("golden_all/block_00_layer00/Mam_X_Proj_FP.txt",      goldfp_xproj);
        $readmemh("golden_all/block_00_layer00/Mam_Delta_FP.txt",       goldfp_delta);
        $readmemh("golden_all/block_00_layer00/Mam_H_State_FP.txt",     gold_h);
        $readmemh("golden_all/block_00_layer00/Mam_Y_Gated_FP.txt",     gold_y_gated);
        $readmemh("golden_all/block_00_layer00/Mam_OutProj_FP.txt",     gold_mout);
        $readmemh("golden_all/block_00_layer00/Final_ITM_Full_FP.txt",  goldfp_final);
        $readmemh("golden_all/block_00_layer00/Mam_W_Norm.txt",         f_Norm_W);

        $display(">> Sanity check block 0:");
        sanity_check(BLK_T, BLK_CH_M, BLK_CH_OUT);
        $display(">> DMA loading block 0...");
        load_weights_and_input;
        $display("   DMA load complete.");

        $display(">> Running block 0...");
        run_one_block;
        $display(">> Comparing block 0...");
        reset_err; compare_all_stages(BLK_T, BLK_CH_OUT, BLK_CH_M);
        print_block_report(0, BLK_T, BLK_CH_OUT, BLK_CH_M);
        accum_err;

        // ??????????????????????????????????????????????????????????????????
        // BLOCK 1  block_01_layer01   T=1000  d_in=64  d_inner=128
        // ??????????????????????????????????????????????????????????????????
        blk = 1; set_block_params(blk);
        $display("\n>> [BLOCK 1] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
                 BLK_T, BLK_CH_OUT*16, BLK_D_INNER, BLK_DT_RANK);

        $readmemh("golden_all/block_01_layer01/P1_Input_X.txt",         f_Xin);
        $readmemh("golden_all/block_01_layer01/P1_Weight_Fused.txt",    f_Wp1);
        $readmemh("golden_all/block_01_layer01/P1_Bias_Fused.txt",      f_Bconv);
        $readmemh("golden_all/block_01_layer01/W_Bot.txt",              f_Wbot);
        $readmemh("golden_all/block_01_layer01/W_B1.txt",               f_Wb1);
        $readmemh("golden_all/block_01_layer01/W_B2.txt",               f_Wb2);
        $readmemh("golden_all/block_01_layer01/W_B3.txt",               f_Wb3);
        $readmemh("golden_all/block_01_layer01/W_B4.txt",               f_Wb4);
        $readmemh("golden_all/block_01_layer01/Inc_BN_Scale.txt",       f_Inc_Scale);
        $readmemh("golden_all/block_01_layer01/Inc_BN_Shift.txt",       f_Inc_Shift);
        $readmemh("golden_all/block_01_layer01/Mam_W_InProj_X.txt",     f_Wmx);
        $readmemh("golden_all/block_01_layer01/Mam_W_InProj_Z.txt",     f_Wmz);
        $readmemh("golden_all/block_01_layer01/Mam_W_Conv.txt",         f_Wconv);
        $readmemh("golden_all/block_01_layer01/Mam_B_Conv.txt",         f_Bconv_dw);
        $readmemh("golden_all/block_01_layer01/Mam_W_XProj.txt",        f_Wxproj);
        $readmemh("golden_all/block_01_layer01/Mam_W_DtProj.txt",       f_Wdt);
        $readmemh("golden_all/block_01_layer01/Mam_B_DtProj.txt",       f_Bdt);
        $readmemh("golden_all/block_01_layer01/Mam_A_signed.txt",       f_Alog);
        $readmemh("golden_all/block_01_layer01/Mam_D_param.txt",        f_Dparam);
        $readmemh("golden_all/block_01_layer01/Mam_W_OutProj.txt",      f_Woutproj);
        $readmemh("golden_all/block_01_layer01/P1_Output_Golden_FP.txt",goldfp_p1);
        $readmemh("golden_all/block_01_layer01/Out_Bot_FP.txt",         goldfp_bot);
        $readmemh("golden_all/block_01_layer01/Out_B1_FP.txt",          goldfp_b1);
        $readmemh("golden_all/block_01_layer01/Out_B2_FP.txt",          goldfp_b2);
        $readmemh("golden_all/block_01_layer01/Out_B3_FP.txt",          goldfp_b3);
        $readmemh("golden_all/block_01_layer01/Out_B4_FP.txt",          goldfp_b4);
        $readmemh("golden_all/block_01_layer01/Mam_Z_Gate_FP.txt",      goldfp_zgate);
        $readmemh("golden_all/block_01_layer01/Mam_U_Silu_FP.txt",      goldfp_usilu);
        $readmemh("golden_all/block_01_layer01/Mam_X_Proj_FP.txt",      goldfp_xproj);
        $readmemh("golden_all/block_01_layer01/Mam_Delta_FP.txt",       goldfp_delta);
        $readmemh("golden_all/block_01_layer01/Mam_H_State_FP.txt",     gold_h);
        $readmemh("golden_all/block_01_layer01/Mam_Y_Gated_FP.txt",     gold_y_gated);
        $readmemh("golden_all/block_01_layer01/Mam_OutProj_FP.txt",     gold_mout);
        $readmemh("golden_all/block_01_layer01/Final_ITM_Full_FP.txt",  goldfp_final);
        $readmemh("golden_all/block_01_layer01/Mam_W_Norm.txt",         f_Norm_W);

        $display(">> Sanity check block 1:");
        sanity_check(BLK_T, BLK_CH_M, BLK_CH_OUT);
        $display(">> Chaining: copying block 0 final output to input...");
        copy_final_to_input(1000, 4);
        $display(">> DMA loading block 1 weights...");
        load_weights_only;
        $display("   DMA load complete.");
        $display(">> Running block 1...");
        run_one_block;
        $display(">> Comparing block 1...");
        reset_err; compare_all_stages(BLK_T, BLK_CH_OUT, BLK_CH_M);
        print_block_report(1, BLK_T, BLK_CH_OUT, BLK_CH_M);
        accum_err;

        // MaxPool after block 1 (T: 1000?500)
        $display("\n>> MaxPool after block 1...");
        maxpool_tb(1000, 4);

        // ??????????????????????????????????????????????????????????????????
        // BLOCK 2  block_02_layer03   T=500  d_in=64  d_inner=128
        // ??????????????????????????????????????????????????????????????????
        blk = 2; set_block_params(blk);
        $display("\n>> [BLOCK 2] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
                 BLK_T, BLK_CH_OUT*16, BLK_D_INNER, BLK_DT_RANK);

        $readmemh("golden_all/block_02_layer03/P1_Input_X.txt",         f_Xin);
        $readmemh("golden_all/block_02_layer03/P1_Weight_Fused.txt",    f_Wp1);
        $readmemh("golden_all/block_02_layer03/P1_Bias_Fused.txt",      f_Bconv);
        $readmemh("golden_all/block_02_layer03/W_Bot.txt",              f_Wbot);
        $readmemh("golden_all/block_02_layer03/W_B1.txt",               f_Wb1);
        $readmemh("golden_all/block_02_layer03/W_B2.txt",               f_Wb2);
        $readmemh("golden_all/block_02_layer03/W_B3.txt",               f_Wb3);
        $readmemh("golden_all/block_02_layer03/W_B4.txt",               f_Wb4);
        $readmemh("golden_all/block_02_layer03/Inc_BN_Scale.txt",       f_Inc_Scale);
        $readmemh("golden_all/block_02_layer03/Inc_BN_Shift.txt",       f_Inc_Shift);
        $readmemh("golden_all/block_02_layer03/Mam_W_InProj_X.txt",     f_Wmx);
        $readmemh("golden_all/block_02_layer03/Mam_W_InProj_Z.txt",     f_Wmz);
        $readmemh("golden_all/block_02_layer03/Mam_W_Conv.txt",         f_Wconv);
        $readmemh("golden_all/block_02_layer03/Mam_B_Conv.txt",         f_Bconv_dw);
        $readmemh("golden_all/block_02_layer03/Mam_W_XProj.txt",        f_Wxproj);
        $readmemh("golden_all/block_02_layer03/Mam_W_DtProj.txt",       f_Wdt);
        $readmemh("golden_all/block_02_layer03/Mam_B_DtProj.txt",       f_Bdt);
        $readmemh("golden_all/block_02_layer03/Mam_A_signed.txt",       f_Alog);
        $readmemh("golden_all/block_02_layer03/Mam_D_param.txt",        f_Dparam);
        $readmemh("golden_all/block_02_layer03/Mam_W_OutProj.txt",      f_Woutproj);
        $readmemh("golden_all/block_02_layer03/P1_Output_Golden_FP.txt",goldfp_p1);
        $readmemh("golden_all/block_02_layer03/Out_Bot_FP.txt",         goldfp_bot);
        $readmemh("golden_all/block_02_layer03/Out_B1_FP.txt",          goldfp_b1);
        $readmemh("golden_all/block_02_layer03/Out_B2_FP.txt",          goldfp_b2);
        $readmemh("golden_all/block_02_layer03/Out_B3_FP.txt",          goldfp_b3);
        $readmemh("golden_all/block_02_layer03/Out_B4_FP.txt",          goldfp_b4);
        $readmemh("golden_all/block_02_layer03/Mam_Z_Gate_FP.txt",      goldfp_zgate);
        $readmemh("golden_all/block_02_layer03/Mam_U_Silu_FP.txt",      goldfp_usilu);
        $readmemh("golden_all/block_02_layer03/Mam_X_Proj_FP.txt",      goldfp_xproj);
        $readmemh("golden_all/block_02_layer03/Mam_Delta_FP.txt",       goldfp_delta);
        $readmemh("golden_all/block_02_layer03/Mam_H_State_FP.txt",     gold_h);
        $readmemh("golden_all/block_02_layer03/Mam_Y_Gated_FP.txt",     gold_y_gated);
        $readmemh("golden_all/block_02_layer03/Mam_OutProj_FP.txt",     gold_mout);
        $readmemh("golden_all/block_02_layer03/Final_ITM_Full_FP.txt",  goldfp_final);
        $readmemh("golden_all/block_02_layer03/Mam_W_Norm.txt",         f_Norm_W);

        $display(">> Sanity check block 2:");
        sanity_check(BLK_T, BLK_CH_M, BLK_CH_OUT);
        $display(">> DMA loading block 2 weights (input from MaxPool of block 1)...");
        load_weights_only;
        $display("   DMA load complete.");
        $display(">> Running block 2...");
        run_one_block;
        $display(">> Comparing block 2...");
        reset_err; compare_all_stages(BLK_T, BLK_CH_OUT, BLK_CH_M);
        print_block_report(2, BLK_T, BLK_CH_OUT, BLK_CH_M);
        accum_err;

        // ??????????????????????????????????????????????????????????????????
        // BLOCK 3  block_03_layer04   T=500  d_in=64  d_inner=128
        // ??????????????????????????????????????????????????????????????????
        blk = 3; set_block_params(blk);
        $display("\n>> [BLOCK 3] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
                 BLK_T, BLK_CH_OUT*16, BLK_D_INNER, BLK_DT_RANK);

        $readmemh("golden_all/block_03_layer04/P1_Input_X.txt",         f_Xin);
        $readmemh("golden_all/block_03_layer04/P1_Weight_Fused.txt",    f_Wp1);
        $readmemh("golden_all/block_03_layer04/P1_Bias_Fused.txt",      f_Bconv);
        $readmemh("golden_all/block_03_layer04/W_Bot.txt",              f_Wbot);
        $readmemh("golden_all/block_03_layer04/W_B1.txt",               f_Wb1);
        $readmemh("golden_all/block_03_layer04/W_B2.txt",               f_Wb2);
        $readmemh("golden_all/block_03_layer04/W_B3.txt",               f_Wb3);
        $readmemh("golden_all/block_03_layer04/W_B4.txt",               f_Wb4);
        $readmemh("golden_all/block_03_layer04/Inc_BN_Scale.txt",       f_Inc_Scale);
        $readmemh("golden_all/block_03_layer04/Inc_BN_Shift.txt",       f_Inc_Shift);
        $readmemh("golden_all/block_03_layer04/Mam_W_InProj_X.txt",     f_Wmx);
        $readmemh("golden_all/block_03_layer04/Mam_W_InProj_Z.txt",     f_Wmz);
        $readmemh("golden_all/block_03_layer04/Mam_W_Conv.txt",         f_Wconv);
        $readmemh("golden_all/block_03_layer04/Mam_B_Conv.txt",         f_Bconv_dw);
        $readmemh("golden_all/block_03_layer04/Mam_W_XProj.txt",        f_Wxproj);
        $readmemh("golden_all/block_03_layer04/Mam_W_DtProj.txt",       f_Wdt);
        $readmemh("golden_all/block_03_layer04/Mam_B_DtProj.txt",       f_Bdt);
        $readmemh("golden_all/block_03_layer04/Mam_A_signed.txt",       f_Alog);
        $readmemh("golden_all/block_03_layer04/Mam_D_param.txt",        f_Dparam);
        $readmemh("golden_all/block_03_layer04/Mam_W_OutProj.txt",      f_Woutproj);
        $readmemh("golden_all/block_03_layer04/P1_Output_Golden_FP.txt",goldfp_p1);
        $readmemh("golden_all/block_03_layer04/Out_Bot_FP.txt",         goldfp_bot);
        $readmemh("golden_all/block_03_layer04/Out_B1_FP.txt",          goldfp_b1);
        $readmemh("golden_all/block_03_layer04/Out_B2_FP.txt",          goldfp_b2);
        $readmemh("golden_all/block_03_layer04/Out_B3_FP.txt",          goldfp_b3);
        $readmemh("golden_all/block_03_layer04/Out_B4_FP.txt",          goldfp_b4);
        $readmemh("golden_all/block_03_layer04/Mam_Z_Gate_FP.txt",      goldfp_zgate);
        $readmemh("golden_all/block_03_layer04/Mam_U_Silu_FP.txt",      goldfp_usilu);
        $readmemh("golden_all/block_03_layer04/Mam_X_Proj_FP.txt",      goldfp_xproj);
        $readmemh("golden_all/block_03_layer04/Mam_Delta_FP.txt",       goldfp_delta);
        $readmemh("golden_all/block_03_layer04/Mam_H_State_FP.txt",     gold_h);
        $readmemh("golden_all/block_03_layer04/Mam_Y_Gated_FP.txt",     gold_y_gated);
        $readmemh("golden_all/block_03_layer04/Mam_OutProj_FP.txt",     gold_mout);
        $readmemh("golden_all/block_03_layer04/Final_ITM_Full_FP.txt",  goldfp_final);
        $readmemh("golden_all/block_03_layer04/Mam_W_Norm.txt",         f_Norm_W);

        $display(">> Sanity check block 3:");
        sanity_check(BLK_T, BLK_CH_M, BLK_CH_OUT);
        $display(">> Chaining: copying block 2 final output to input...");
        copy_final_to_input(500, 4);
        $display(">> DMA loading block 3 weights...");
        load_weights_only;
        $display("   DMA load complete.");
        $display(">> Running block 3...");
        run_one_block;
        $display(">> Comparing block 3...");
        reset_err; compare_all_stages(BLK_T, BLK_CH_OUT, BLK_CH_M);
        print_block_report(3, BLK_T, BLK_CH_OUT, BLK_CH_M);
        accum_err;

        // MaxPool after block 3 (T: 500?250)
        $display("\n>> MaxPool after block 3...");
        maxpool_tb(500, 4);

        // ??????????????????????????????????????????????????????????????????
        // BLOCK 4  block_04_layer06   T=250  d_in=128  d_inner=256
        // ??????????????????????????????????????????????????????????????????
        blk = 4; set_block_params(blk);
        $display("\n>> [BLOCK 4] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
                 BLK_T, BLK_CH_IN*16, BLK_D_INNER, BLK_DT_RANK);

        $readmemh("golden_all/block_04_layer06/P1_Input_X.txt",         f_Xin);
        $readmemh("golden_all/block_04_layer06/P1_Weight_Fused.txt",    f_Wp1);
        $readmemh("golden_all/block_04_layer06/P1_Bias_Fused.txt",      f_Bconv);
        $readmemh("golden_all/block_04_layer06/W_Bot.txt",              f_Wbot);
        $readmemh("golden_all/block_04_layer06/W_B1.txt",               f_Wb1);
        $readmemh("golden_all/block_04_layer06/W_B2.txt",               f_Wb2);
        $readmemh("golden_all/block_04_layer06/W_B3.txt",               f_Wb3);
        $readmemh("golden_all/block_04_layer06/W_B4.txt",               f_Wb4);
        $readmemh("golden_all/block_04_layer06/Inc_BN_Scale.txt",       f_Inc_Scale);
        $readmemh("golden_all/block_04_layer06/Inc_BN_Shift.txt",       f_Inc_Shift);
        $readmemh("golden_all/block_04_layer06/Mam_W_InProj_X.txt",     f_Wmx);
        $readmemh("golden_all/block_04_layer06/Mam_W_InProj_Z.txt",     f_Wmz);
        $readmemh("golden_all/block_04_layer06/Mam_W_Conv.txt",         f_Wconv);
        $readmemh("golden_all/block_04_layer06/Mam_B_Conv.txt",         f_Bconv_dw);
        $readmemh("golden_all/block_04_layer06/Mam_W_XProj.txt",        f_Wxproj);
        $readmemh("golden_all/block_04_layer06/Mam_W_DtProj.txt",       f_Wdt);
        $readmemh("golden_all/block_04_layer06/Mam_B_DtProj.txt",       f_Bdt);
        $readmemh("golden_all/block_04_layer06/Mam_A_signed.txt",       f_Alog);
        $readmemh("golden_all/block_04_layer06/Mam_D_param.txt",        f_Dparam);
        $readmemh("golden_all/block_04_layer06/Mam_W_OutProj.txt",      f_Woutproj);
        $readmemh("golden_all/block_04_layer06/P1_Output_Golden_FP.txt",goldfp_p1);
        $readmemh("golden_all/block_04_layer06/Out_Bot_FP.txt",         goldfp_bot);
        $readmemh("golden_all/block_04_layer06/Out_B1_FP.txt",          goldfp_b1);
        $readmemh("golden_all/block_04_layer06/Out_B2_FP.txt",          goldfp_b2);
        $readmemh("golden_all/block_04_layer06/Out_B3_FP.txt",          goldfp_b3);
        $readmemh("golden_all/block_04_layer06/Out_B4_FP.txt",          goldfp_b4);
        $readmemh("golden_all/block_04_layer06/Mam_Z_Gate_FP.txt",      goldfp_zgate);
        $readmemh("golden_all/block_04_layer06/Mam_U_Silu_FP.txt",      goldfp_usilu);
        $readmemh("golden_all/block_04_layer06/Mam_X_Proj_FP.txt",      goldfp_xproj);
        $readmemh("golden_all/block_04_layer06/Mam_Delta_FP.txt",       goldfp_delta);
        $readmemh("golden_all/block_04_layer06/Mam_H_State_FP.txt",     gold_h);
        $readmemh("golden_all/block_04_layer06/Mam_Y_Gated_FP.txt",     gold_y_gated);
        $readmemh("golden_all/block_04_layer06/Mam_OutProj_FP.txt",     gold_mout);
        $readmemh("golden_all/block_04_layer06/Final_ITM_Full_FP.txt",  goldfp_final);
        $readmemh("golden_all/block_04_layer06/Mam_W_Norm.txt",         f_Norm_W);

        $display(">> Sanity check block 4:");
        sanity_check(BLK_T, BLK_CH_M, BLK_CH_OUT);
        $display(">> DMA loading block 4 weights (input from MaxPool of block 3)...");
        load_weights_only;
        $display("   DMA load complete.");
        $display(">> Running block 4...");
        run_one_block;
        $display(">> Comparing block 4...");
        reset_err; compare_all_stages(BLK_T, BLK_CH_OUT, BLK_CH_M);
        print_block_report(4, BLK_T, BLK_CH_OUT, BLK_CH_M);
        accum_err;

        // ??????????????????????????????????????????????????????????????????
        // FINAL SUMMARY - ALL 5 BLOCKS
        // ??????????????????????????????????????????????????????????????????
        $display("\n");
        $display("################################################################");
        $display("#                                                              #");
        $display("#   ITMN V9 - FINAL SUMMARY  (5 blocks)                       #");
        $display("#                                                              #");
        $display("################################################################");
        $display("  FRAC_BITS=%0d   TOLERANCE=%0d", `FRAC_BITS, TOLERANCE);
        $display("  Total simulation cycles  : %0d", total_cycles_all);
        $display("  Estimated @ 100 MHz      : %.3f ms",
                 1.0 * total_cycles_all * 10 / 1_000_000);
        $display("  ??????????????????????????????????????????????????????????");
        $display("  Stage (5-block total)   | TotalErr | WorstMax | Overall");
        $display("  ??????????????????????? +??????????+??????????+?????????");
        $display("  P1 Output               |  %7d |    %5d  | %s",
                 tot_err_p1,  worst_p1,   (tot_err_p1==0)  ?"PASS":"FAIL");
        $display("  Mam Z_Gate  (M1b)       |  %7d |    (N/A) | %s",
                 tot_err_zg,              (tot_err_zg==0)  ?"PASS":"FAIL");
        $display("  Mam U_Safe  (M3_COPY)   |  %7d |    (N/A) | %s",
                 tot_err_us,              (tot_err_us==0)  ?"PASS":"FAIL");
        $display("  Mam X_Proj  (M4)        |  %7d |    (N/A) | %s",
                 tot_err_xp,              (tot_err_xp==0)  ?"PASS":"FAIL");
        $display("  Mam Delta   (M5)        |  %7d |    (N/A) | %s",
                 tot_err_dt,              (tot_err_dt==0)  ?"PASS":"FAIL");
        $display("  Mam H_State (M6a)       |  %7d |    %5d  | %s",
                 tot_err_h,   worst_h,    (tot_err_h==0)   ?"PASS":"FAIL");
        $display("  Mam Y_Gated (M7)        |  %7d |    (N/A) | %s",
                 tot_err_yg,              (tot_err_yg==0)  ?"PASS":"FAIL");
        $display("  Mam OutProj (M8)        |  %7d |    %5d  | %s",
                 tot_err_mout,worst_mout, (tot_err_mout==0)?"PASS":"FAIL");
        $display("  ??????????????????????? +??????????+??????????+?????????");
        $display("  Final Full Output       |  %7d |    %5d  | %s",
                 tot_err_fin, worst_fin,  (tot_err_fin==0) ?"PASS":"FAIL");
        $display("################################################################");
        $display("");
        if (tot_err_fin==0 && tot_err_mout==0 && tot_err_h==0 &&
            tot_err_p1==0  && tot_err_yg==0)
            $display("  *** ALL STAGES PASS - ITMN V9 HW-EXACT VERIFIED ***");
        else
            $display("  *** ERRORS DETECTED - see per-block reports above ***");
        $display("");
        $display("  GOAL: all err=0, max_d=0 across all 5 blocks.");
        $display("################################################################");
        $stop;
    end

endmodule