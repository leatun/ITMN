`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_5BLOCK — Complete 5-block Mamba sweep with per-block DMA reload.
//
// Flow: for bk in 0..4:
//   1. $readmemh block bk's per-tensor goldens into TB memories.
//   2. Set runtime dims (CH_OUT/CH_M/DT_RANK/T_MAX).
//   3. DMA-preload weights + consts + input into ram_weight/ram_const/ram_main.
//      (All weights preload permanently — streaming removed 2026-07-13.)
//   4. Assert start. Record cyc_start.
//   5. Wait done_all.
//   6. Record cyc_end. Deassert start.
//
// Final report:
//   Per-block: preload cycles, compute cycles, cyc/timestep.
//   Grand total = sum of all preload+compute across 5 blocks.
//
// Dims table (from _block_params.v / itmn_pipeline.py):
//   B0/B1: d_model=64  d_inner=128 d_state=16 dt_rank=4 T=1000
//   B2/B3: d_model=64  d_inner=128 d_state=16 dt_rank=4 T=500
//   B4:    d_model=128 d_inner=256 d_state=16 dt_rank=8 T=250
// ============================================================================

module tb_Mamba_Top_5BLOCK;

    // ---- Universal storage sized for worst-case B4 ----
    reg [15:0] wx_mem   [0:32767];    // max D_INNER * D_MODEL = 256*128
    reg [15:0] wz_mem   [0:32767];
    reg [15:0] wo_mem   [0:32767];    // max D_MODEL * D_INNER
    reg [15:0] wdw_mem  [0:1023];     // max D_INNER * 4 = 256*4
    reg [15:0] wxp_mem  [0:12287];    // max N_PAD * D_INNER = 48*256
    reg [15:0] wdt_mem  [0:2047];     // max D_INNER * DT_RANK = 256*8
    reg [15:0] wA_mem   [0:4095];     // max D_INNER * D_STATE = 256*16
    reg [15:0] gam_mem  [0:127];      // max D_MODEL = 128
    reg [15:0] bdw_mem  [0:255];      // max D_INNER = 256
    reg [15:0] bdt_mem  [0:255];
    reg [15:0] Dp_mem   [0:255];
    reg [15:0] xn_mem   [0:63999];    // max D_MODEL * T_TOT = 64*1000

    // ---- Current-block dims (updated per invocation) ----
    integer cur_D_MODEL, cur_D_INNER, cur_D_STATE, cur_DT_RANK, cur_N_PAD;
    integer cur_T, cur_T_TOT;

    // ---- Runtime ports ----
    reg [3:0]  CH_OUT, CH_M, DT_RANK;
    reg [9:0]  T_MAX;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg         start = 0;
    wire        done_stage, done_all;
    reg  [3:0]  run_stage = 4'd0;

    reg          dma_write_en = 0;
    reg  [1:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    reg          dma_read_en  = 0;
    reg  [1:0]   dma_rtarget  = 0;
    reg  [14:0]  dma_raddr    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top dut (
        .clk(clk), .rst(rst), .start(start),
        .done_stage(done_stage), .done_all(done_all),
        .run_stage(run_stage), .T_MAX(T_MAX),
        .CH_OUT(CH_OUT), .CH_M(CH_M), .DT_RANK(DT_RANK),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // ---- Result tracking ----
    integer preload_start [0:4];
    integer preload_end   [0:4];
    integer cyc_start_arr [0:4];
    integer cyc_end_arr   [0:4];
    integer bk_it;

    // ---- Loop / packing scratch ----
    integer c_out_grp, c_grp_out, c_in, k, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp;

    task dma_wr;
        input [1:0]   target;
        input [14:0]  addr;
        input [255:0] data;
        begin
            @(negedge clk);
            dma_write_en = 1;
            dma_target   = target;
            dma_addr     = addr;
            dma_wdata    = data;
        end
    endtask

    // ---- Load per-block goldens + set dims ----
    task load_block;
        input integer bk;
        begin
            case (bk)
                0: begin
                    $readmemh("golden_all/block_00_layer00/Mam_W_InProj_X.txt", wx_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_W_InProj_Z.txt", wz_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_W_OutProj.txt", wo_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_W_Conv.txt",    wdw_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_W_XProj.txt",   wxp_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_W_DtProj.txt",  wdt_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_A_signed.txt",  wA_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_W_Norm.txt",    gam_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_B_Conv.txt",    bdw_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_B_DtProj.txt",  bdt_mem);
                    $readmemh("golden_all/block_00_layer00/Mam_D_param.txt",   Dp_mem);
                    $readmemh("golden_all/block_00_layer00/P1_Output_Golden_FP.txt", xn_mem);
                    cur_D_MODEL=64; cur_D_INNER=128; cur_D_STATE=16;
                    cur_DT_RANK=4;  cur_N_PAD=48;    cur_T=1000; cur_T_TOT=1000;
                    CH_OUT=4'd4; CH_M=4'd8; DT_RANK=4'd4; T_MAX=10'd1000;
                end
                1: begin
                    $readmemh("golden_all/block_01_layer01/Mam_W_InProj_X.txt", wx_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_W_InProj_Z.txt", wz_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_W_OutProj.txt", wo_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_W_Conv.txt",    wdw_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_W_XProj.txt",   wxp_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_W_DtProj.txt",  wdt_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_A_signed.txt",  wA_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_W_Norm.txt",    gam_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_B_Conv.txt",    bdw_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_B_DtProj.txt",  bdt_mem);
                    $readmemh("golden_all/block_01_layer01/Mam_D_param.txt",   Dp_mem);
                    $readmemh("golden_all/block_01_layer01/P1_Output_Golden_FP.txt", xn_mem);
                    cur_D_MODEL=64; cur_D_INNER=128; cur_D_STATE=16;
                    cur_DT_RANK=4;  cur_N_PAD=48;    cur_T=1000; cur_T_TOT=1000;
                    CH_OUT=4'd4; CH_M=4'd8; DT_RANK=4'd4; T_MAX=10'd1000;
                end
                2: begin
                    $readmemh("golden_all/block_02_layer03/Mam_W_InProj_X.txt", wx_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_W_InProj_Z.txt", wz_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_W_OutProj.txt", wo_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_W_Conv.txt",    wdw_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_W_XProj.txt",   wxp_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_W_DtProj.txt",  wdt_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_A_signed.txt",  wA_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_W_Norm.txt",    gam_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_B_Conv.txt",    bdw_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_B_DtProj.txt",  bdt_mem);
                    $readmemh("golden_all/block_02_layer03/Mam_D_param.txt",   Dp_mem);
                    $readmemh("golden_all/block_02_layer03/P1_Output_Golden_FP.txt", xn_mem);
                    cur_D_MODEL=64; cur_D_INNER=128; cur_D_STATE=16;
                    cur_DT_RANK=4;  cur_N_PAD=48;    cur_T=500;  cur_T_TOT=500;
                    CH_OUT=4'd4; CH_M=4'd8; DT_RANK=4'd4; T_MAX=10'd500;
                end
                3: begin
                    $readmemh("golden_all/block_03_layer04/Mam_W_InProj_X.txt", wx_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_W_InProj_Z.txt", wz_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_W_OutProj.txt", wo_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_W_Conv.txt",    wdw_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_W_XProj.txt",   wxp_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_W_DtProj.txt",  wdt_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_A_signed.txt",  wA_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_W_Norm.txt",    gam_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_B_Conv.txt",    bdw_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_B_DtProj.txt",  bdt_mem);
                    $readmemh("golden_all/block_03_layer04/Mam_D_param.txt",   Dp_mem);
                    $readmemh("golden_all/block_03_layer04/P1_Output_Golden_FP.txt", xn_mem);
                    cur_D_MODEL=64; cur_D_INNER=128; cur_D_STATE=16;
                    cur_DT_RANK=4;  cur_N_PAD=48;    cur_T=500;  cur_T_TOT=500;
                    CH_OUT=4'd4; CH_M=4'd8; DT_RANK=4'd4; T_MAX=10'd500;
                end
                4: begin
                    $readmemh("golden_all/block_04_layer06/Mam_W_InProj_X.txt", wx_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_W_InProj_Z.txt", wz_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_W_OutProj.txt", wo_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_W_Conv.txt",    wdw_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_W_XProj.txt",   wxp_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_W_DtProj.txt",  wdt_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_A_signed.txt",  wA_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_W_Norm.txt",    gam_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_B_Conv.txt",    bdw_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_B_DtProj.txt",  bdt_mem);
                    $readmemh("golden_all/block_04_layer06/Mam_D_param.txt",   Dp_mem);
                    $readmemh("golden_all/block_04_layer06/P1_Output_Golden_FP.txt", xn_mem);
                    cur_D_MODEL=128; cur_D_INNER=256; cur_D_STATE=16;
                    cur_DT_RANK=8;   cur_N_PAD=48;    cur_T=250;  cur_T_TOT=250;
                    CH_OUT=4'd8; CH_M=4'd0; DT_RANK=4'd8; T_MAX=10'd250;
                end
            endcase
        end
    endtask

    // ---- DMA-preload current block's weights + consts + input ----
    task preload_block;
        begin
            // W_InProj_X → SLOT_A
            for (c_out_grp=0; c_out_grp<(cur_D_INNER/16); c_out_grp=c_out_grp+1)
                for (c_in=0; c_in<cur_D_MODEL; c_in=c_in+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = wx_mem[(c_out_grp*16+lane)*cur_D_MODEL + c_in];
                    dma_wr(2'd2, `W_INPROJ_X_BASE + c_out_grp*cur_D_MODEL + c_in, word_tmp);
                end

            // W_InProj_Z → SLOT_B
            for (c_out_grp=0; c_out_grp<(cur_D_INNER/16); c_out_grp=c_out_grp+1)
                for (c_in=0; c_in<cur_D_MODEL; c_in=c_in+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = wz_mem[(c_out_grp*16+lane)*cur_D_MODEL + c_in];
                    dma_wr(2'd2, `W_INPROJ_Z_BASE + c_out_grp*cur_D_MODEL + c_in, word_tmp);
                end

            // W_OutProj → permanent region (post-refactor: all blocks preload)
            for (c_out_grp=0; c_out_grp<(cur_D_MODEL/16); c_out_grp=c_out_grp+1)
                for (c_in=0; c_in<cur_D_INNER; c_in=c_in+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = wo_mem[(c_out_grp*16+lane)*cur_D_INNER + c_in];
                    dma_wr(2'd2, `W_OUTPROJ_BASE + c_out_grp*cur_D_INNER + c_in, word_tmp);
                end

            // W_DW
            for (c_grp_in=0; c_grp_in<(cur_D_INNER/16); c_grp_in=c_grp_in+1)
                for (k=0; k<4; k=k+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = wdw_mem[(c_grp_in*16+lane)*4 + k];
                    dma_wr(2'd2, `W_DW_BASE + c_grp_in*4 + k, word_tmp);
                end

            // W_XProj
            for (c_grp_out=0; c_grp_out<(cur_N_PAD/16); c_grp_out=c_grp_out+1)
                for (c_in=0; c_in<cur_D_INNER; c_in=c_in+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = wxp_mem[(c_grp_out*16+lane)*cur_D_INNER + c_in];
                    dma_wr(2'd2, `W_XPROJ_BASE + c_grp_out*cur_D_INNER + c_in, word_tmp);
                end

            // W_DtProj
            for (c_grp_out=0; c_grp_out<(cur_D_INNER/16); c_grp_out=c_grp_out+1)
                for (k=0; k<cur_DT_RANK; k=k+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = wdt_mem[(c_grp_out*16+lane)*cur_DT_RANK + k];
                    dma_wr(2'd2, `W_DTPROJ_BASE + c_grp_out*cur_DT_RANK + k, word_tmp);
                end

            // W_A (1 word per channel)
            for (c_in=0; c_in<cur_D_INNER; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<cur_D_STATE; lane=lane+1)
                    word_tmp[lane*16+:16] = wA_mem[c_in*cur_D_STATE + lane];
                dma_wr(2'd2, `W_A_BASE + c_in, word_tmp);
            end

            // ---- Constants ----
            for (c_grp_in=0; c_grp_in<(cur_D_MODEL/16); c_grp_in=c_grp_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = gam_mem[c_grp_in*16 + lane];
                dma_wr(2'd3, `C_W_NORM_BASE + c_grp_in, word_tmp);
            end
            for (c_grp_in=0; c_grp_in<(cur_D_INNER/16); c_grp_in=c_grp_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = bdw_mem[c_grp_in*16 + lane];
                dma_wr(2'd3, `C_B_DW_BASE + c_grp_in, word_tmp);
            end
            for (c_grp_in=0; c_grp_in<(cur_D_INNER/16); c_grp_in=c_grp_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = bdt_mem[c_grp_in*16 + lane];
                dma_wr(2'd3, `C_B_DT_BASE + c_grp_in, word_tmp);
            end
            for (c_grp_in=0; c_grp_in<(cur_D_INNER/16); c_grp_in=c_grp_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = Dp_mem[c_grp_in*16 + lane];
                dma_wr(2'd3, `C_D_PARAM_BASE + c_grp_in, word_tmp);
            end

            // ---- INPUT ----
            for (t_cur=0; t_cur<cur_T; t_cur=t_cur+1)
                for (c_grp_in=0; c_grp_in<(cur_D_MODEL/16); c_grp_in=c_grp_in+1) begin
                    word_tmp = 256'b0;
                    for (lane=0; lane<16; lane=lane+1)
                        word_tmp[lane*16+:16] = xn_mem[(c_grp_in*16+lane)*cur_T_TOT + t_cur];
                    dma_wr(2'd0, `PT_INPUT + t_cur*(cur_D_MODEL/16) + c_grp_in, word_tmp);
                end
            @(negedge clk); dma_write_en = 0;
        end
    endtask

    task wait_done;
        begin
            wait (done_all == 1'b1);
        end
    endtask

    // ---- Main flow ----
    initial begin
        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        for (bk_it = 0; bk_it < 5; bk_it = bk_it + 1) begin
            $display("");
            $display("======== BLOCK %0d ========", bk_it);
            preload_start[bk_it] = $time / 10;
            load_block(bk_it);
            preload_block;
            preload_end[bk_it] = $time / 10;
            $display("  preload cycles = %0d",
                     preload_end[bk_it] - preload_start[bk_it]);

            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            cyc_start_arr[bk_it] = $time / 10;
            $display("  start cyc = %0d", cyc_start_arr[bk_it]);

            wait_done;

            cyc_end_arr[bk_it] = $time / 10;
            $display("  end cyc   = %0d  compute = %0d cycles",
                     cyc_end_arr[bk_it],
                     cyc_end_arr[bk_it] - cyc_start_arr[bk_it]);

            // Idle a few cycles before next block preload
            @(negedge clk); @(negedge clk); @(negedge clk);
        end

        // ---- Final report ----
        $display("");
        $display("========== 5-BLOCK CYCLE SUMMARY ==========");
        $display("| Blk |  T   | Preload | Compute  |  cyc/t |");
        $display("|-----|------|---------|----------|--------|");
        $display("| B0  | 1000 | %7d | %8d | %6d |",
                 preload_end[0] - preload_start[0],
                 cyc_end_arr[0] - cyc_start_arr[0],
                 (cyc_end_arr[0] - cyc_start_arr[0]) / 1000);
        $display("| B1  | 1000 | %7d | %8d | %6d |",
                 preload_end[1] - preload_start[1],
                 cyc_end_arr[1] - cyc_start_arr[1],
                 (cyc_end_arr[1] - cyc_start_arr[1]) / 1000);
        $display("| B2  |  500 | %7d | %8d | %6d |",
                 preload_end[2] - preload_start[2],
                 cyc_end_arr[2] - cyc_start_arr[2],
                 (cyc_end_arr[2] - cyc_start_arr[2]) / 500);
        $display("| B3  |  500 | %7d | %8d | %6d |",
                 preload_end[3] - preload_start[3],
                 cyc_end_arr[3] - cyc_start_arr[3],
                 (cyc_end_arr[3] - cyc_start_arr[3]) / 500);
        $display("| B4  |  250 | %7d | %8d | %6d |",
                 preload_end[4] - preload_start[4],
                 cyc_end_arr[4] - cyc_start_arr[4],
                 (cyc_end_arr[4] - cyc_start_arr[4]) / 250);
        $display("===========================================");
        $display("Grand total (preload+compute across 5 blocks) = %0d cycles",
                 cyc_end_arr[4] - preload_start[0]);
        $display("===== TB 5-BLOCK DONE =====");
        $finish;
    end

    initial begin
        #10000000000;
        $display("ERROR: timeout at %0t", $time);
        $finish;
    end

endmodule
