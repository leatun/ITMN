`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_M8 — byte-exact end-to-end test of M8 stage on Mamba_Top.
//
// Block 0 (T=1000, d_inner=128, d_out=64):
//   1. DMA load W_OutProj → ram_weight (target=2)
//   2. DMA load Y_Gated   → ram_a       (target=0)
//   3. Pulse start
//   4. Wait done_m8
//   5. DMA read ram_b mamba_out (target=1), unpack, compare with Mam_OutProj_FP
//
// Goldens loaded ($readmemh from sim CWD):
//   Mam_W_OutProj.txt     (d_out × d_inner = 64 × 128 = 8192 vals)
//   Mam_Y_Gated_FP.txt    (d_inner × T     = 128 × 1000 = 128000 vals)
//   Mam_OutProj_FP.txt    (d_out × T       = 64 × 1000 = 64000 vals, expected)
//
// Packing convention:
//   Weight word at addr (c_out_grp * d_inner + c_in) packs
//     W[c_out_grp*16+0..+15, c_in]  (lane order: lane k = c_out = c_out_grp*16+k)
//   Y_gated word at addr (t * (d_inner/16) + c_grp_in) packs
//     y[c_grp_in*16+0..+15, t]
//   Mamba_out word at addr (t * (d_out/16) + c_out_grp) packs
//     mamba_out[c_out_grp*16+0..+15, t]
//
// File index convention (row-major in itmn_pipeline.save_iq):
//   W:     index = c_out * d_inner + c_in
//   Y:     index = c * T + t
//   Mout:  index = c_out * T + t
//
// Test scope: first T_TEST timesteps. T_TEST=4 → 256 compares. Bump to 1000
// for full sweep (64000 compares).
// ============================================================================

module tb_Mamba_Top_M8;

    // ----- Dimensions (block 0) — centralised in _parameter.v -----
    localparam D_IN_INNER = `B0_D_INNER;
    localparam D_OUT      = `B0_D_MODEL;
    localparam T_TOT      = `B0_T_TOT;
    localparam T_TEST     = 1000;

    // ----- Clock / reset -----
    reg clk = 0;
    reg rst = 1;
    always #5 clk = ~clk;

    // ----- Layout (must match Mamba_Top) -----
    localparam A_Y_GATED_BASE   = `A_Y_GATED_BASE;
    localparam W_OUTPROJ_BASE   = `W_OUTPROJ_BASE;
    localparam B_MAMBA_OUT_BASE = `B_MAMBA_OUT_BASE;

    // ----- DUT control -----
    reg         start = 0;
    wire        done_stage;
    wire        done_all;

    reg  [3:0]  run_stage = 4'd8;    // M8
    reg  [9:0]  T_MAX  = T_TEST;     // sweep only T_TEST timesteps
    reg  [3:0]  CH_OUT = 4'd4;       // d_model=64 → 64/16=4
    reg  [3:0]  CH_M    = 4'd8;       // d_inner=128 → 128/16=8
    reg  [3:0]  DT_RANK = 4'd4;       // unused for M8 but required port

    // ----- DMA wires -----
    reg          dma_write_en = 0;
    reg  [1:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    reg          dma_read_en  = 0;
    reg  [1:0]   dma_rtarget  = 0;
    reg  [14:0]  dma_raddr    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top dut (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .done_stage   (done_stage),
        .done_all     (done_all),
        .run_stage    (run_stage),
        .T_MAX        (T_MAX),
        .CH_OUT       (CH_OUT),
        .CH_M         (CH_M),
        .DT_RANK      (DT_RANK),
        .dma_write_en (dma_write_en),
        .dma_target   (dma_target),
        .dma_addr     (dma_addr),
        .dma_wdata    (dma_wdata),
        .dma_read_en  (dma_read_en),
        .dma_rtarget  (dma_rtarget),
        .dma_raddr    (dma_raddr),
        .dma_rdata    (dma_rdata)
    );

    // ----- Golden storage -----
    reg [15:0] w_mem   [0:D_OUT*D_IN_INNER-1];     // (d_out, d_inner)
    reg [15:0] y_mem   [0:D_IN_INNER*T_TOT-1];     // (d_inner, T)
    reg [15:0] exp_mem [0:D_OUT*T_TOT-1];          // (d_out, T) expected

    integer errors   = 0;
    integer compares = 0;
    integer i, c_out_grp, c_in, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp;
    reg [255:0] readback;
    reg signed [15:0] got_val, exp_val;

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_W_OutProj.txt",  w_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Y_Gated_FP.txt", y_mem);
        $readmemh("golden_all/block_00_layer00/Mam_OutProj_FP.txt", exp_mem);

        // ---- Reset ----
        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // ----------------------------------------------------------
        // 1) DMA load W_OutProj into ram_weight (target=2)
        //    Pack per (c_out_grp, c_in) → 1 word of 16 weights
        // ----------------------------------------------------------
        $display("[DMA] Loading W_OutProj (%0d words)...", (D_OUT/16) * D_IN_INNER);
        for (c_out_grp = 0; c_out_grp < (D_OUT/16); c_out_grp = c_out_grp + 1) begin
            for (c_in = 0; c_in < D_IN_INNER; c_in = c_in + 1) begin
                // Build packed word
                word_tmp = 256'b0;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    word_tmp[lane*16 +: 16] =
                        w_mem[(c_out_grp*16 + lane)*D_IN_INNER + c_in];
                end
                @(negedge clk);
                dma_write_en = 1;
                dma_target   = 2'd2;             // ram_weight
                dma_addr     = W_OUTPROJ_BASE + c_out_grp * D_IN_INNER + c_in;
                dma_wdata    = word_tmp;
            end
        end
        @(negedge clk); dma_write_en = 0;

        // ----------------------------------------------------------
        // 2) DMA load Y_Gated into ram_a (target=0)
        //    Pack per (t, c_grp_in) → 1 word of 16 channels
        // ----------------------------------------------------------
        $display("[DMA] Loading Y_Gated (%0d words)...", T_TEST * (D_IN_INNER/16));
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_grp_in = 0; c_grp_in < (D_IN_INNER/16); c_grp_in = c_grp_in + 1) begin
                word_tmp = 256'b0;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    word_tmp[lane*16 +: 16] =
                        y_mem[(c_grp_in*16 + lane)*T_TOT + t_cur];
                end
                @(negedge clk);
                dma_write_en = 1;
                dma_target   = 2'd0;             // ram_a
                dma_addr     = A_Y_GATED_BASE + t_cur * (D_IN_INNER/16) + c_grp_in;
                dma_wdata    = word_tmp;
            end
        end
        @(negedge clk); dma_write_en = 0;

        // ----------------------------------------------------------
        // 3) Pulse start, wait done
        // ----------------------------------------------------------
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        $display("[FSM] M8 running...");

        // Wait for done_stage
        wait (done_stage == 1'b1);
        $display("[FSM] done_stage asserted at time %0t", $time);

        // ----------------------------------------------------------
        // 4) DMA-read ram_b mamba_out, unpack, compare
        // ----------------------------------------------------------
        @(negedge clk);
        dma_read_en = 1;
        dma_rtarget = 2'd1;                       // ram_b

        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_out_grp = 0; c_out_grp < (D_OUT/16); c_out_grp = c_out_grp + 1) begin
                @(negedge clk);
                dma_raddr = B_MAMBA_OUT_BASE + t_cur * (D_OUT/16) + c_out_grp;
                @(posedge clk);
                @(negedge clk);    // 1-cycle BRAM read latency: dma_rdata valid now
                readback = dma_rdata;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    got_val = readback[lane*16 +: 16];
                    exp_val = exp_mem[(c_out_grp*16 + lane)*T_TOT + t_cur];
                    compares = compares + 1;
                    if (got_val !== exp_val) begin
                        $display("FAIL t=%0d c_out=%0d  got=%6d (0x%04h)  exp=%6d (0x%04h)",
                                 t_cur, c_out_grp*16 + lane,
                                 got_val, got_val & 16'hFFFF,
                                 exp_val, exp_val & 16'hFFFF);
                        errors = errors + 1;
                    end
                end
            end
        end
        dma_read_en = 0;

        $display("");
        $display("---- tb_Mamba_Top_M8 summary ----");
        $display("  timesteps tested : %0d / %0d", T_TEST, T_TOT);
        $display("  total compares   : %0d", compares);
        $display("  errors           : %0d", errors);
        if (errors == 0)
            $display("===== TB M8 BYTE-EXACT PASS =====");
        else
            $display("===== TB M8 FAIL =====");
        $finish;
    end

    // Watchdog
    initial begin
        #50000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
