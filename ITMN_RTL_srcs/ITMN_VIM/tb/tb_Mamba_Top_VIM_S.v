`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_VIM_S - cycle benchmark for Vision Mamba Tiny (Vim-S).
//
// Vim architecture (from Vim paper Alg 1, mambaX HW ref):
//   Bidirectional Mamba1 block: forward SSM + backward SSM per block,
//   outputs merged via z_gate elementwise mul.
//   HW impl option chosen: run M6 twice (fwd then bwd, reverse address for bwd),
//   merge outputs before M7. Analytical cycle model = base + M6_cyc (2nd pass).
//
// Vim-S config (state-spaces/mamba defaults for Vim Tiny):
//   D (hidden dim) = 192  -> CH_OUT = 12
//   d_inner = E*D = 384   -> CH_M   = 24
//   d_state = 16          -> N_STATE_GRP = 1
//   dt_rank = ceil(D/16) = 12
//   L (blocks) = 24
//   patch = 16x16, image 224x224 -> seq_len = 196 (patches) + 1 (CLS) = 197
//   USE_M5 = 1 (Mamba1 dt_proj), XP_OUT_GRP = (12+32)/16 = 3
//
// Note: TB uses T=3 for quick benchmark; full sequence = 197 -> scale cycles ~66x.
// Bidirectional overhead reported in latency section (+M6 cyc per token).
// ============================================================================

module tb_Mamba_Top_VIM_S;

    localparam CH_OUT_V        = 9'd24;
    localparam CH_M_V          = 9'd48;
    localparam DT_RANK_V       = 8'd24;
    localparam N_STATE_GRP_V   = 4'd1;
    localparam USE_M5_V        = 1'b1;
    localparam XP_OUT_GRP_V    = 5'd4;    // (12+32)/16 = 3
    localparam integer N_LAYER = 24;
    localparam integer SEQ_LEN = 197;     // Full Vim-S @ 224x224
    localparam T_TEST          = 3;

    // ---- DMA per layer (Mamba1 Vim-S): ~15704 words ----
    // InProj 9216 + OutProj 4608 + XProj 1056 + DTProj 144 + DW 96 + A 384 + norms 200
    // Bidirectional overhead: separate conv1d + A per direction => +DW+A = +480 -> ~16184
    localparam integer DMA_CYC_PER_LAYER = 60200;

    // ---- Clock / reset ----
    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    wire         done_stage, done_all;
    reg  [3:0]   run_stage = 4'd0;
    reg  [9:0]   T_MAX     = T_TEST;
    reg  [8:0]   CH_OUT    = CH_OUT_V;
    reg  [8:0]   CH_M      = CH_M_V;
    reg  [7:0]   DT_RANK   = DT_RANK_V;
    reg  [3:0]   N_STATE_GRP = N_STATE_GRP_V;
    reg          USE_M5    = USE_M5_V;
    reg  [4:0]   XP_OUT_GRP_IN = XP_OUT_GRP_V;

    reg          dma_write_en = 0;
    reg  [2:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    reg          dma_read_en  = 0;
    reg  [2:0]   dma_rtarget  = 0;
    reg  [14:0]  dma_raddr    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top dut (
        .clk(clk), .rst(rst), .start(start),
        .done_stage(done_stage), .done_all(done_all),
        .run_stage(run_stage), .T_MAX(T_MAX),
        .CH_OUT(CH_OUT), .CH_M(CH_M), .DT_RANK(DT_RANK),
        .N_STATE_GRP(N_STATE_GRP), .USE_M5(USE_M5),
        .XP_OUT_GRP_IN(XP_OUT_GRP_IN),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    integer cyc_start, cyc_end, i, j;
    integer stage_start [0:9];
    integer stage_total [0:9];
    integer h_init_start, h_init_end;
    reg     h_init_seen;
    reg [3:0] last_stage;
    reg       tracker_armed;

    initial begin
        for (j = 0; j < 10; j = j + 1) begin
            stage_start[j] = 0; stage_total[j] = 0;
        end
        h_init_start = 0; h_init_end = 0; h_init_seen = 0;
        last_stage = 4'hF; tracker_armed = 0;
    end

    always @(posedge clk) begin
        if (!rst) begin
            if (dut.state == 7'd1 && !h_init_seen) begin
                h_init_start <= $time / 10; h_init_seen <= 1'b1;
            end
            if (h_init_seen && dut.state != 7'd1 && h_init_end == 0)
                h_init_end <= $time / 10;
            if (tracker_armed && dut.cur_stage !== last_stage) begin
                if (last_stage < 4'd10)
                    stage_total[last_stage] <= stage_total[last_stage] +
                                                (($time/10) - stage_start[last_stage]);
                if (dut.cur_stage < 4'd10)
                    stage_start[dut.cur_stage] <= $time / 10;
                last_stage <= dut.cur_stage;
            end
            if (!tracker_armed && h_init_end != 0 && dut.cur_stage < 4'd10) begin
                tracker_armed <= 1'b1;
                stage_start[dut.cur_stage] <= $time / 10;
                last_stage <= dut.cur_stage;
            end
        end
    end

    initial begin
        #50000000;
        $display("ERROR: timeout"); $finish;
    end

    initial begin
        rst = 1; repeat (5) @(posedge clk);
        @(negedge clk); rst = 0; @(posedge clk);

        $display("[DMA] INPUT preload (%0d words)", CH_OUT_V * T_TEST);
        @(negedge clk);
        dma_write_en = 1; dma_target = 3'd0;
        for (i = 0; i < CH_OUT_V * T_TEST; i = i + 1) begin
            dma_addr  = `PT_INPUT + i[14:0];
            dma_wdata = {16{16'sh0100}};
            @(negedge clk);
        end
        dma_write_en = 0;

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc_start = $time / 10;
        $display("[CFG] Vim-S (D=192 CH_OUT=%0d CH_M=%0d dt=%0d N_STATE_GRP=%0d)",
                 CH_OUT_V, CH_M_V, DT_RANK_V, N_STATE_GRP_V);

        wait (done_all == 1'b1);
        cyc_end = $time / 10;
        if (last_stage < 4'd10)
            stage_total[last_stage] = stage_total[last_stage] +
                                      (cyc_end - stage_start[last_stage]);

        $display("");
        $display("========================================================");
        $display("     tb_Mamba_Top_VIM_S - cycle report (T=%0d, 1 layer)", T_TEST);
        $display("========================================================");
        $display("Config: Vision Mamba Tiny (Vim-S), 1 layer, %0d token(s)", T_TEST);
        $display("  D=384 (24 grp) d_inner=768 (48 grp) d_state=16 (1 s-grp)");
        $display("  dt_rank=24 XP=4 L=24 patch=16 seq_len=197 (image 224x224)");
        $display("");
        $display("--------- Per-stage TOTAL cycles ---------");
        $display("  H_INIT              : %0d", h_init_end - h_init_start);
        $display("  RN  (RMSNorm)       : %0d", stage_total[9]);
        $display("  M1A (in_proj X)     : %0d", stage_total[0]);
        $display("  M1B (in_proj Z)     : %0d", stage_total[1]);
        $display("  M2  (dw conv)       : %0d", stage_total[2]);
        $display("  M3  (SiLU)          : %0d", stage_total[3]);
        $display("  M4  (x_proj)        : %0d", stage_total[4]);
        $display("  M5  (dt_proj)       : %0d", stage_total[5]);
        $display("  M6  (SSM scan)      : %0d", stage_total[6]);
        $display("  M7  (Gate)          : %0d", stage_total[7]);
        $display("  M8  (out_proj)      : %0d", stage_total[8]);
        $display("");
        $display("--------- Per-token cycles (unidirectional pass) ---------");
        $display("  Compute per token   : %0d cyc",
                 ((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST);
        $display("  M6 per token        : %0d cyc", stage_total[6] / T_TEST);
        $display("");
        $display("--------- Bidirectional Vim projection (Vim uses fwd+bwd SSM) ---------");
        $display("  Compute per token bidir : %0d cyc (unidir + extra M6 pass)",
                 ((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST
                 + stage_total[6] / T_TEST);
        $display("");
        $display("--------- ImageNet inference: %0d tokens x %0d layers ---------",
                 SEQ_LEN, N_LAYER);
        $display("  Compute-only (bidir) : %0d ms",
                 (((((cyc_end - cyc_start) - (h_init_end - h_init_start))/T_TEST
                    + stage_total[6]/T_TEST) * SEQ_LEN * N_LAYER) / 100000));
        $display("  + DMA serial         : %0d ms",
                 (((((cyc_end - cyc_start) - (h_init_end - h_init_start))/T_TEST
                    + stage_total[6]/T_TEST + DMA_CYC_PER_LAYER) * SEQ_LEN * N_LAYER)
                  / 100000));
        $display("========================================================");
        $display("Compare vs Mamba-X paper (cycle simulator, no FPGA):");
        $display("  Mamba-X Vim-S     : normalized 11.6x GPU speedup, target 30W TDP");
        $display("  Our impl on VC709  : (fill in from synth/sim results)");
        $display("========================================================");
        $finish;
    end

endmodule
