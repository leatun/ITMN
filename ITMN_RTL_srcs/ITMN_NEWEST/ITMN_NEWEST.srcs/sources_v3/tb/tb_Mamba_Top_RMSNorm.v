`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_RMSNorm — byte-exact test of RMSNorm stage on Mamba_Top.
//
// Block 0 (T=1000, d_model=64):
//   out[c, t] = sat16(sat16(x[c,t] * gamma[c] >> 11) * S_t >> 11)
//   where S_t = RSqrt_ROM[clip(sum_d[t] >> 21, 0, 8191)]
//         sum_d[t] = Σ_c x[c,t]² (full int40 precision)
//
// Flow:
//   1. DMA load gamma           → ram_const @ C_W_NORM_BASE (=0)
//   2. DMA load P1_Output_Golden → ram_b     @ B_INPUT_BASE (=0)
//   3. start with run_stage=4'd9; wait done_stage
//   4. DMA read ram_a @ A_X_NORM_BASE (=0); compare with P1_Norm_Output_FP
//
// Goldens needed:
//   Mam_W_Norm.txt              (d_model = 64 vals)
//   P1_Output_Golden_FP.txt     (d_model × T = 64000 vals) — input
//   P1_Norm_Output_FP.txt       (d_model × T = 64000 vals) — expected output
//
// **Lưu ý**: P1_Norm_Output_FP.txt có thể không có sẵn trong project golden_all
// (extract cũ). Re-run itmn_pipeline.py extract trong WSL nếu thiếu.
//
// Packing convention:
//   gamma word @ addr c_grp holds w_norm[c_grp*16+0..+15]   (d_model/16 = 4 words for B0)
//   x_word    @ addr (t * d_model/16 + c_grp) holds x[c_grp*16+0..+15, t]
//   out_word  @ addr (t * d_model/16 + c_grp) holds out[c_grp*16+0..+15, t]
//
// File row-major index: gamma[c], x[c*T + t], out[c*T + t]
//
// Test scope: T_TEST timesteps × d_model channels.
// Default T_TEST=4 → 256 compares.
// ============================================================================

module tb_Mamba_Top_RMSNorm;

    localparam D_MODEL = `B0_D_MODEL;
    localparam T_TOT   = `B0_T_TOT;
    localparam T_TEST  = 1;

    localparam C_W_NORM_BASE = `C_W_NORM_BASE;
    localparam B_INPUT_BASE  = `B_INPUT_BASE;
    localparam A_X_NORM_BASE = `A_X_NORM_BASE;

    reg clk = 0;
    reg rst = 1;
    always #5 clk = ~clk;

    reg         start = 0;
    wire        done_stage;
    wire        done_all;
    reg  [3:0]  run_stage = 4'd9;
    reg  [9:0]  T_MAX     = T_TEST;
    reg  [3:0]  CH_OUT    = 4'd4;
    reg  [3:0]  CH_M      = 4'd8;
    reg  [3:0]  DT_RANK   = 4'd4;

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

    reg [15:0] gamma_mem [0:D_MODEL-1];           // (d_model,)
    reg [15:0] x_mem     [0:D_MODEL*T_TOT-1];     // P1_Output: (d_model, T)
    reg [15:0] exp_mem   [0:D_MODEL*T_TOT-1];     // P1_Norm_Output: (d_model, T)

    integer errors   = 0;
    integer compares = 0;
    integer i, c_grp, t_cur, lane;
    reg [255:0] word_tmp;
    reg [255:0] readback;
    reg signed [15:0] got_val, exp_val;

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
        $readmemh("golden_all/block_00_layer00/Mam_W_Norm.txt",          gamma_mem);
        $readmemh("golden_all/block_00_layer00/P1_Output_Golden_FP.txt", x_mem);
        $readmemh("golden_all/block_00_layer00/P1_Norm_Output_FP.txt",   exp_mem);

        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // 1) Load gamma into ram_const (target=3)
        //    addr = c_grp, word = pack 16 channels
        $display("[DMA] Loading gamma (%0d words)...", D_MODEL/16);
        for (c_grp = 0; c_grp < (D_MODEL/16); c_grp = c_grp + 1) begin
            word_tmp = 256'b0;
            for (lane = 0; lane < 16; lane = lane + 1)
                word_tmp[lane*16 +: 16] = gamma_mem[c_grp*16 + lane];
            dma_wr(2'd3, C_W_NORM_BASE + c_grp, word_tmp);
        end

        // 2) Load P1_Output into ram_b (target=1)
        //    addr = t * d_model/16 + c_grp, word = pack 16 channels at t
        $display("[DMA] Loading P1_Output (%0d words)...", T_TEST * (D_MODEL/16));
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_grp = 0; c_grp < (D_MODEL/16); c_grp = c_grp + 1) begin
                word_tmp = 256'b0;
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = x_mem[(c_grp*16 + lane)*T_TOT + t_cur];
                dma_wr(2'd1, B_INPUT_BASE + t_cur * (D_MODEL/16) + c_grp, word_tmp);
            end
        end
        @(negedge clk); dma_write_en = 0;

        // 3) Run RMSNorm
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        $display("[FSM] RMSNorm running...");
        wait (done_stage == 1'b1);
        $display("[FSM] done_stage asserted at time %0t", $time);

        @(negedge clk);
        wait (done_stage == 1'b0);

        // 4) Readback ram_a (target=0) and compare
        @(negedge clk);
        dma_read_en = 1;
        dma_rtarget = 2'd0;
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_grp = 0; c_grp < (D_MODEL/16); c_grp = c_grp + 1) begin
                @(negedge clk);
                dma_raddr = A_X_NORM_BASE + t_cur * (D_MODEL/16) + c_grp;
                @(posedge clk);
                @(negedge clk);
                readback = dma_rdata;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    got_val = readback[lane*16 +: 16];
                    exp_val = exp_mem[(c_grp*16 + lane)*T_TOT + t_cur];
                    compares = compares + 1;
                    if (got_val !== exp_val) begin
//                        if (errors < 1000) begin
                            $display("FAIL t=%0d c=%0d  got=%6d (0x%04h)  exp=%6d (0x%04h)",
                                     t_cur, c_grp*16 + lane,
                                     got_val, got_val & 16'hFFFF,
                                     exp_val, exp_val & 16'hFFFF);
//                        end       
                        errors = errors + 1;
                    end
                    else 
                    $display("pass t=%0d c=%0d  got=%6d (0x%04h)  exp=%6d (0x%04h)",
                                     t_cur, c_grp*16 + lane,
                                     got_val, got_val & 16'hFFFF,
                                     exp_val, exp_val & 16'hFFFF);
                end
            end
        end
        dma_read_en = 0;

        $display("");
        $display("---- tb_Mamba_Top_RMSNorm summary ----");
        $display("  timesteps tested : %0d / %0d", T_TEST, T_TOT);
        $display("  total compares   : %0d", compares);
        $display("  errors           : %0d", errors);
        if (errors == 0)
            $display("===== TB RMSNorm BYTE-EXACT PASS =====");
        else
            $display("===== TB RMSNorm FAIL =====");
        $finish;
    end

    initial begin
        #50000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
