`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_M1AB — byte-exact test of M1A + M1B stages on Mamba_Top.
//
// Block 0 reference (T=1000, d_model=64, d_inner=128):
//   M1A: x_inner[t, co] = sat16(Σ_ci W_X[co, ci] · x_norm[t, ci] >> 11)
//   M1B: z_gate [t, co] = sat16(Σ_ci W_Z[co, ci] · x_norm[t, ci] >> 11)
//
// Flow:
//   1. DMA load W_InProj_X → ram_weight @ W_INPROJ_X_BASE (=0)
//   2. DMA load W_InProj_Z → ram_weight @ W_INPROJ_Z_BASE (=512)
//   3. DMA load X_NORM     → ram_a      @ A_X_NORM_BASE   (=0)
//   4. start with run_stage=4'd0 (M1A); wait done_stage
//   5. DMA read ram_b @ B_X_INNER_BASE  (=0), compare with Mam_X_Inner_FP
//   6. start with run_stage=4'd1 (M1B); wait done_stage
//   7. DMA read ram_b @ B_Z_GATE_BASE   (=8000), compare with Mam_Z_Gate_FP
//
// Goldens needed (from itmn_pipeline.py extract):
//   Mam_W_InProj_X.txt       (d_inner × d_model = 128 × 64 = 8192 vals)
//   Mam_W_InProj_Z.txt       (d_inner × d_model = 128 × 64 = 8192 vals)
//   P1_Norm_Output_FP.txt    (d_model × T = 64 × 1000 = 64000 vals) — input
//   Mam_X_Inner_FP.txt       (d_inner × T = 128 × 1000 = 128000 vals) — expected M1A
//   Mam_Z_Gate_FP.txt        (d_inner × T = 128 × 1000 = 128000 vals) — expected M1B
//
// Packing convention (matches Mamba_Top layout):
//   X_NORM word @ addr(t * d_model/16 + c_grp_in) holds x_norm[c_grp_in*16+0..+15, t]
//   W_InProj_X/Z word @ addr(c_out_grp * d_model + c_in) holds W[c_out_grp*16+0..+15, c_in]
//   X_INNER / Z_GATE word @ addr(t * d_inner/16 + c_out_grp) holds out[c_out_grp*16+0..+15, t]
//
// File index convention (row-major in itmn_pipeline.save_iq):
//   X_NORM:   index = c * T + t
//   W:        index = c_out * d_model + c_in
//   X_INNER:  index = c_out * T + t
//   Z_GATE:   index = c_out * T + t
//
// Test scope: T_TEST timesteps. Default 4 → 4×128×2 = 1024 compares.
// Bump T_TEST=1000 for full sweep (256K compares, ~10s sim).
// ============================================================================

module tb_Mamba_Top_M1AB;

    // ----- Dimensions (block 0) — centralised in _parameter.v -----
    localparam D_MODEL = `B0_D_MODEL;
    localparam D_INNER = `B0_D_INNER;
    localparam T_TOT   = `B0_T_TOT;
    localparam T_TEST  = 1000;

    localparam W_INPROJ_X_BASE = `W_INPROJ_X_BASE;
    localparam W_INPROJ_Z_BASE = `W_INPROJ_Z_BASE;
    localparam B_X_INNER_BASE  = `B_X_INNER_BASE;
    localparam B_Z_GATE_BASE   = `B_Z_GATE_BASE;

    // ----- Clock / reset -----
    reg clk = 0;
    reg rst = 1;
    always #5 clk = ~clk;

    // ----- DUT control -----
    reg         start = 0;
    wire        done_stage;
    wire        done_all;
    reg  [3:0]  run_stage = 4'd0;
    reg  [9:0]  T_MAX     = T_TEST;
    reg  [3:0]  CH_OUT    = 4'd4;       // d_model=64 → 64/16=4
    reg  [3:0]  CH_M      = 4'd8;       // d_inner=128 → 128/16=8
    reg  [3:0]  DT_RANK   = 4'd4;

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
    reg [15:0] wx_mem   [0:D_INNER*D_MODEL-1];     // W_InProj_X (d_inner, d_model)
    reg [15:0] wz_mem   [0:D_INNER*D_MODEL-1];     // W_InProj_Z
    reg [15:0] xn_mem   [0:D_MODEL*T_TOT-1];       // X_NORM (d_model, T) — input
    reg [15:0] xi_mem   [0:D_INNER*T_TOT-1];       // expected X_INNER (d_inner, T)
    reg [15:0] zg_mem   [0:D_INNER*T_TOT-1];       // expected Z_GATE (d_inner, T)

    integer errors_m1a   = 0;
    integer errors_m1b   = 0;
    integer compares_m1a = 0;
    integer compares_m1b = 0;
    integer i, c_out_grp, c_in, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp;
    reg [255:0] readback;
    reg signed [15:0] got_val, exp_val;

    // ----- Helper: DMA write 1 word -----
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

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_W_InProj_X.txt",   wx_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_InProj_Z.txt",   wz_mem);
        $readmemh("golden_all/block_00_layer00/P1_Norm_Output_FP.txt", xn_mem);
        $readmemh("golden_all/block_00_layer00/Mam_X_Inner_FP.txt",   xi_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Z_Gate_FP.txt",    zg_mem);

        // ---- Reset ----
        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // ----------------------------------------------------------
        // 1) DMA load W_InProj_X
        //    Pack per (c_out_grp, c_in) → 1 word of 16 c_out
        //    addr = W_INPROJ_X_BASE + c_out_grp * d_model + c_in
        // ----------------------------------------------------------
        $display("[DMA] Loading W_InProj_X (%0d words)...", (D_INNER/16) * D_MODEL);
        for (c_out_grp = 0; c_out_grp < (D_INNER/16); c_out_grp = c_out_grp + 1) begin
            for (c_in = 0; c_in < D_MODEL; c_in = c_in + 1) begin
                word_tmp = 256'b0;
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] =
                        wx_mem[(c_out_grp*16 + lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_X_BASE + c_out_grp * D_MODEL + c_in, word_tmp);
            end
        end

        // 2) DMA load W_InProj_Z
        $display("[DMA] Loading W_InProj_Z (%0d words)...", (D_INNER/16) * D_MODEL);
        for (c_out_grp = 0; c_out_grp < (D_INNER/16); c_out_grp = c_out_grp + 1) begin
            for (c_in = 0; c_in < D_MODEL; c_in = c_in + 1) begin
                word_tmp = 256'b0;
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] =
                        wz_mem[(c_out_grp*16 + lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_Z_BASE + c_out_grp * D_MODEL + c_in, word_tmp);
            end
        end

        // 3) DMA load X_NORM
        //    Pack per (t, c_grp_in) → 1 word of 16 channels
        //    addr = A_X_NORM_BASE + t * d_model/16 + c_grp_in
        $display("[DMA] Loading X_NORM (%0d words)...", T_TEST * (D_MODEL/16));
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_grp_in = 0; c_grp_in < (D_MODEL/16); c_grp_in = c_grp_in + 1) begin
                word_tmp = 256'b0;
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] =
                        xn_mem[(c_grp_in*16 + lane)*T_TOT + t_cur];
                dma_wr(2'd0, t_cur * (D_MODEL/16) + c_grp_in, word_tmp);
            end
        end
        @(negedge clk); dma_write_en = 0;

        // ----------------------------------------------------------
        // M1A run + check
        // ----------------------------------------------------------
        @(negedge clk); run_stage = 4'd0; start = 1;
        @(negedge clk); start = 0;
        $display("[FSM] M1A running...");
        wait (done_stage == 1'b1);
        $display("[FSM] M1A done at time %0t", $time);

        // Wait a bit for done_stage to clear after !start
        @(negedge clk);
        wait (done_stage == 1'b0);

        // Readback ram_b @ B_X_INNER_BASE
        @(negedge clk);
        dma_read_en = 1;
        dma_rtarget = 2'd1;
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_out_grp = 0; c_out_grp < (D_INNER/16); c_out_grp = c_out_grp + 1) begin
                @(negedge clk);
                dma_raddr = B_X_INNER_BASE + t_cur * (D_INNER/16) + c_out_grp;
                @(posedge clk);
                @(negedge clk);
                readback = dma_rdata;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    got_val = readback[lane*16 +: 16];
                    exp_val = xi_mem[(c_out_grp*16 + lane)*T_TOT + t_cur];
                    compares_m1a = compares_m1a + 1;
                    if (got_val !== exp_val) begin
                        $display("M1A FAIL t=%0d c_out=%0d  got=%6d (0x%04h)  exp=%6d (0x%04h)",
                                 t_cur, c_out_grp*16 + lane,
                                 got_val, got_val & 16'hFFFF,
                                 exp_val, exp_val & 16'hFFFF);
                        errors_m1a = errors_m1a + 1;
                    end
                end
            end
        end
        dma_read_en = 0;

        // ----------------------------------------------------------
        // M1B run + check
        // ----------------------------------------------------------
        @(negedge clk); run_stage = 4'd1; start = 1;
        @(negedge clk); start = 0;
        $display("[FSM] M1B running...");
        wait (done_stage == 1'b1);
        $display("[FSM] M1B done at time %0t", $time);

        @(negedge clk);
        wait (done_stage == 1'b0);

        // Readback ram_b @ B_Z_GATE_BASE
        @(negedge clk);
        dma_read_en = 1;
        dma_rtarget = 2'd1;
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_out_grp = 0; c_out_grp < (D_INNER/16); c_out_grp = c_out_grp + 1) begin
                @(negedge clk);
                dma_raddr = B_Z_GATE_BASE + t_cur * (D_INNER/16) + c_out_grp;
                @(posedge clk);
                @(negedge clk);
                readback = dma_rdata;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    got_val = readback[lane*16 +: 16];
                    exp_val = zg_mem[(c_out_grp*16 + lane)*T_TOT + t_cur];
                    compares_m1b = compares_m1b + 1;
                    if (got_val !== exp_val) begin
                        $display("M1B FAIL t=%0d c_out=%0d  got=%6d (0x%04h)  exp=%6d (0x%04h)",
                                 t_cur, c_out_grp*16 + lane,
                                 got_val, got_val & 16'hFFFF,
                                 exp_val, exp_val & 16'hFFFF);
                        errors_m1b = errors_m1b + 1;
                    end
                end
            end
        end
        dma_read_en = 0;

        // ----------------------------------------------------------
        // Summary
        // ----------------------------------------------------------
        $display("");
        $display("---- tb_Mamba_Top_M1AB summary ----");
        $display("  timesteps tested : %0d / %0d", T_TEST, T_TOT);
        $display("  M1A compares     : %0d   errors: %0d", compares_m1a, errors_m1a);
        $display("  M1B compares     : %0d   errors: %0d", compares_m1b, errors_m1b);
        if (errors_m1a == 0 && errors_m1b == 0)
            $display("===== TB M1A+M1B BYTE-EXACT PASS =====");
        else
            $display("===== TB M1A+M1B FAIL =====");
        $finish;
    end

    initial begin
        #100000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
