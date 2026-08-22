`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_MAMBA1_130M — cycle benchmark for Mamba1-130M (MARCA workload).
//
// Config (Mamba1-130M @ 1 layer, from state-spaces/mamba-130m-hf):
//   d_model  = 768   -> CH_OUT = 48
//   d_inner  = 1536  -> CH_M   = 96
//   d_state  = 16    -> N_STATE_GRP = 1  (Mamba1 default, much smaller than Mamba2)
//   dt_rank  = 48    -> USE_M5 = 1 (Mamba1 uses dt_proj matrix)
//   x_proj outputs: (dt_rank+2*d_state) = 48+32 = 80 elems -> XP_OUT_GRP = 5
//
// Focus: cycle counts. Weights uninitialized (garbage OK).
// ============================================================================

module tb_Mamba_Top_MAMBA1_130M;

    // ---- Mamba1-130M config (MARCA workload) ----
    localparam CH_OUT_V        = 9'd48;
    localparam CH_M_V          = 9'd96;
    localparam DT_RANK_V       = 8'd48;    // Mamba1: dt_rank = ceil(d_model/16)
    localparam N_STATE_GRP_V   = 4'd1;     // Mamba1: d_state=16 -> 1 grp
    localparam USE_M5_V        = 1'b1;     // Mamba1: enable M5 dt-proj stage
    localparam XP_OUT_GRP_V    = 5'd5;     // (48+32)/16 = 5
    localparam integer N_LAYER = 24;
    localparam T_TEST          = 3;

    // ---- DMA reload cost (paper-fair MODEL, per-layer, Mamba1 arch) ----
    //   W_InProj (X+Z):  2*d_inner*d_model/16
    //   W_OutProj:       d_inner*d_model/16
    //   W_XProj:         (dt_rank+2*d_state)*d_inner/16
    //   W_DTProj:        d_model*dt_rank/16  (Mamba1-only)
    //   W_DW:            d_inner*4/16
    //   W_A:             d_inner*d_state/16
    //   Norms + biases:  ~200
    // 130M: 147456+73728+7680+2304+384+1536+200 = 233,288
    localparam integer DMA_CYC_PER_LAYER = 233288;

    // ---- Clock / reset ----
    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;                             // 100 MHz

    // ---- DUT interface ----
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
    reg  [2:0]   dma_target   = 0;   // Widened 2→3-bit for weight bank_B (3'd4)
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

    integer cyc_start, cyc_end;
    integer i;

    // ---- Per-stage cycle accumulators ----
    //  Index = cur_stage encoding: 0=M1A 1=M1B 2=M2 3=M3 4=M4 5=M5 6=M6 7=M7 8=M8 9=RN
    integer stage_start [0:9];
    integer stage_total [0:9];
    integer h_init_start, h_init_end;
    reg     h_init_seen;

    integer j;
    initial begin
        for (j = 0; j < 10; j = j + 1) begin
            stage_start[j] = 0;
            stage_total[j] = 0;
        end
        h_init_start = 0;
        h_init_end   = 0;
        h_init_seen  = 0;
    end

    reg [3:0] last_stage;
    reg       tracker_armed;
    initial begin
        last_stage    = 4'hF;
        tracker_armed = 0;
    end
    always @(posedge clk) begin
        if (!rst) begin
            // H_INIT tracking (S_H_INIT = 7'd1)
            if (dut.state == 7'd1 && !h_init_seen) begin
                h_init_start <= $time / 10;
                h_init_seen  <= 1'b1;
            end
            if (h_init_seen && dut.state != 7'd1 && h_init_end == 0) begin
                h_init_end <= $time / 10;
            end
            // Per-stage tracking: on stage change, close old + open new
            if (tracker_armed && dut.cur_stage !== last_stage) begin
                if (last_stage < 4'd10)
                    stage_total[last_stage] <= stage_total[last_stage] +
                                                (($time/10) - stage_start[last_stage]);
                if (dut.cur_stage < 4'd10)
                    stage_start[dut.cur_stage] <= $time / 10;
                $display("[cyc=%0d] stage %0d → %0d", $time/10, last_stage, dut.cur_stage);
                last_stage <= dut.cur_stage;
            end
            // Arm tracker after H_INIT completes (h_init_end captured → past S_H_INIT)
            if (!tracker_armed && h_init_end != 0 && dut.cur_stage < 4'd10) begin
                tracker_armed     <= 1'b1;
                stage_start[dut.cur_stage] <= $time / 10;
                last_stage        <= dut.cur_stage;
            end
        end
    end

    // ---- Watchdog ----
    initial begin
        #50000000;
        $display("ERROR: timeout (50ms = 5M cyc). last stage=%0d state=%0d ctr_g=%0d ctr_l=%0d ctr_s=%0d",
                 dut.cur_stage, dut.state, dut.ctr_g, dut.ctr_l, dut.ctr_s);
        $display("  Phase 6 debug: c2_state=%0d c2_g=%0d c2_l=%0d c2_s=%0d c2_load=%0d",
                 dut.c2_state, dut.c2_g, dut.c2_l, dut.c2_s, dut.c2_load);
        $display("  c1_writes_main=%0b c2_writes_main=%0b c2_done=%0b",
                 dut.c1_writes_main, dut.c2_writes_main, dut.c2_done);
        $finish;
    end

    // ---- Phase 5 M6 debug: periodic snapshot every 100k cyc during M6 ----
    integer snap_ctr; initial snap_ctr = 0;
    always @(posedge clk) begin
        if (!rst && dut.cur_stage == 4'd6) begin
            snap_ctr <= snap_ctr + 1;
            if (snap_ctr % 100000 == 0)
                $display("[cyc=%0d] SNAP M6: c1_state=%0d ctr_g=%0d/l=%0d/s=%0d  |  c2_state=%0d c2_g=%0d/l=%0d/s=%0d done=%0b",
                         $time/10, dut.state, dut.ctr_g, dut.ctr_l, dut.ctr_s,
                         dut.c2_state, dut.c2_g, dut.c2_l, dut.c2_s, dut.c2_done);
        end
    end

    initial begin
        cyc_start = 0; cyc_end = 0;

        rst = 1;
        repeat (5) @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // Minimal INPUT preload (RMSNorm reads CH_OUT words at t=0)
        $display("[DMA] INPUT preload (%0d words)", CH_OUT_V * T_TEST);
        @(negedge clk);
        dma_write_en = 1;
        dma_target   = 3'd0;   // ram_main (PT_INPUT preload)
        for (i = 0; i < CH_OUT_V * T_TEST; i = i + 1) begin
            dma_addr  = `PT_INPUT + i[14:0];
            dma_wdata = {16{16'sh0100}};
            @(negedge clk);
        end
        dma_write_en = 0;

        // ---- Start pipeline ----
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc_start = $time / 10;
        $display("");
        $display("[CFG] CH_OUT=%0d CH_M=%0d N_STATE_GRP=%0d USE_M5=%0d XP_OUT_GRP=%0d T=%0d",
                 CH_OUT_V, CH_M_V, N_STATE_GRP_V, USE_M5_V, XP_OUT_GRP_V, T_TEST);
        $display("[FSM] start cyc=%0d", cyc_start);

        wait (done_all == 1'b1);
        cyc_end = $time / 10;

        // Close final stage timer
        if (last_stage < 4'd10)
            stage_total[last_stage] = stage_total[last_stage] +
                                       (cyc_end - stage_start[last_stage]);

        // ==== REPORT ====
        $display("");
        $display("========================================================");
        $display("     tb_Mamba_Top_MAMBA1_130M - cycle report (T=%0d tokens, 1 layer)", T_TEST);
        $display("========================================================");
        $display("Config: Mamba1-130M (MARCA workload), 1 layer, %0d token(s)", T_TEST);
        $display("  d_model=768 (48 grp)  d_inner=1536 (96 grp)  d_state=16 (1 s-grp)");
        $display("  dt_rank=48  M4 xp_out_grp=5 (dt+B+C=80/16)  M5 dt_proj enabled");
        $display("");
        $display("--------- Per-stage TOTAL cycles (all %0d tokens) ---------", T_TEST);
        $display("  H_INIT (once, not per-token) : %0d", h_init_end - h_init_start);
        $display("  STG_RN  (RMSNorm)            : %0d", stage_total[9]);
        $display("  STG_M1A (in_proj X)          : %0d", stage_total[0]);
        $display("  STG_M1B (in_proj Z)          : %0d", stage_total[1]);
        $display("  STG_M2  (depthwise conv)     : %0d", stage_total[2]);
        $display("  STG_M3  (SiLU stream)        : %0d", stage_total[3]);
        $display("  STG_M4  (x_proj dt/B/C)      : %0d", stage_total[4]);
        $display("  STG_M5  (dt_proj Mamba1)     : %0d", stage_total[5]);
        $display("  STG_M6  (SSM scan)           : %0d", stage_total[6]);
        $display("  STG_M7  (Gate, merged in M6) : %0d", stage_total[7]);
        $display("  STG_M8  (out_proj)           : %0d", stage_total[8]);
        $display("");
        $display("--------- Per-token AVERAGE cycles ---------");
        $display("  Compute per token (excl H_INIT): %0d cyc",
                 ((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST);
        $display("  M6 per token                   : %0d cyc", stage_total[6] / T_TEST);
        $display("  M1A+M1B+M4+M8 per token        : %0d cyc",
                 (stage_total[0] + stage_total[1] + stage_total[4] + stage_total[8]) / T_TEST);
        $display("");
        $display("--------- Decode latency (autoregressive, 24 layers/token) ---------");
        $display("  Compute-only  : %0d us/tok  (%0d tok/s)",
                 (((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST * N_LAYER) / 100,
                 1000000 / ((((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST * N_LAYER) / 100));
        $display("  + DMA serial  : %0d us/tok  (%0d tok/s)",
                 ((((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST + DMA_CYC_PER_LAYER) * N_LAYER) / 100,
                 1000000 / (((((cyc_end - cyc_start) - (h_init_end - h_init_start)) / T_TEST + DMA_CYC_PER_LAYER) * N_LAYER) / 100));
        $display("  + DMA prefetch: %0d us/tok  (%0d tok/s)",
                 (DMA_CYC_PER_LAYER * N_LAYER) / 100,
                 1000000 / ((DMA_CYC_PER_LAYER * N_LAYER) / 100));
        $display("");
        $display("--------- Paper baseline (MARCA, ICCAD'24, ASIC 28nm scaled) ---------");
        $display("  MARCA Mamba-130M speedup vs GPU : ~194x avg (their Fig 9)");
        $display("  ITMN vs Mamba-CPU/GPU: TBD (compute from cycles above)");
        $display("========================================================");
        $finish;
    end

endmodule
