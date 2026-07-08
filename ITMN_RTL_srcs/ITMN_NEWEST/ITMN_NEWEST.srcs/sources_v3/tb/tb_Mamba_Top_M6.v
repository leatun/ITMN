`timescale 1ns/1ps
`include "_parameter.v"

// tb_Mamba_Top_M6 — byte-exact M6 SSM scan.
//
// Block 0: T=1000, d_inner=128, d_state=16.
//   Inputs (DMA-loaded):
//     U      → ram_a @ A_U_BASE_M6=0           (d_inner × T = 8000 words)
//     DELTA  → ram_a @ A_DELTA_BASE_M6=8000    (8000 words)
//     B      → ram_a @ A_B_BASE_M6=16000       (T words, 1 word/t)
//     C      → ram_a @ A_C_BASE_M6=17000       (T words)
//     A      → ram_w @ W_A_BASE=2400           (d_inner words, 1 word/channel)
//     D      → ram_const @ C_D_PARAM_BASE=40   (d_inner/16 words)
//   Output:
//     Y_SSM  → ram_b @ B_Y_SSM_BASE_M6=0       (d_inner × T words)
//
// Goldens:
//   Mam_U_Silu_FP.txt      (d_inner, T)
//   Mam_Delta_FP.txt       (d_inner, T)
//   Mam_B_Aligned_FP.txt   (d_state, T)
//   Mam_C_Aligned_FP.txt   (d_state, T)
//   Mam_A_signed.txt       (d_inner, d_state)
//   Mam_D_param.txt        (d_inner)
//   Mam_Y_SSM_FP.txt       (d_inner, T) — expected
//
// Note: M6 is sequential in T, ~30 cycles per (c, t) → for B0 T_TEST=2:
//       2 × 128 × 30 ≈ 7700 cycles + overhead. ~80 μs sim.

module tb_Mamba_Top_M6;
    localparam D_INNER  = `B0_D_INNER;
    localparam D_STATE  = `B0_D_STATE;
    localparam T_TOT    = `B0_T_TOT;
    localparam T_TEST   = 1000;
    localparam A_U_BASE     = `A_U_BASE_M6;
    localparam A_DELTA_BASE = `A_DELTA_BASE_M6;
    localparam A_B_BASE     = `A_B_BASE_M6;
    localparam A_C_BASE     = `A_C_BASE_M6;
    localparam W_A_BASE     = `W_A_BASE;
    localparam C_D_BASE     = `C_D_PARAM_BASE;
    localparam B_Y_SSM_BASE = `B_Y_SSM_BASE_M6;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done_stage, done_all;
    reg [3:0] run_stage = 4'd6;
    reg [9:0] T_MAX  = T_TEST;
    reg [3:0] CH_OUT = 4'd4;
    reg [3:0] CH_M   = 4'd8;
    reg [3:0] DT_RANK = 4'd4;

    reg          dma_write_en = 0;
    reg  [1:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    reg          dma_read_en  = 0;
    reg  [1:0]   dma_rtarget  = 0;
    reg  [14:0]  dma_raddr    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top dut (.clk(clk),.rst(rst),.start(start),.done_stage(done_stage),
        .done_all(done_all),.run_stage(run_stage),.T_MAX(T_MAX),.CH_OUT(CH_OUT),
        .CH_M(CH_M),.DT_RANK(DT_RANK),.dma_write_en(dma_write_en),
        .dma_target(dma_target),.dma_addr(dma_addr),.dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en),.dma_rtarget(dma_rtarget),.dma_raddr(dma_raddr),
        .dma_rdata(dma_rdata));

    reg [15:0] u_mem      [0:D_INNER*T_TOT-1];
    reg [15:0] delta_mem  [0:D_INNER*T_TOT-1];
    reg [15:0] B_mem      [0:D_STATE*T_TOT-1];
    reg [15:0] C_mem      [0:D_STATE*T_TOT-1];
    reg [15:0] A_mem      [0:D_INNER*D_STATE-1];
    reg [15:0] D_mem      [0:D_INNER-1];
    reg [15:0] exp_mem    [0:D_INNER*T_TOT-1];

    integer errors=0, compares=0;
    integer c_grp, t_cur, lane, c, s;
    reg [255:0] word_tmp, readback;
    reg signed [15:0] got_val, exp_val;

    task dma_wr(input [1:0] target, input [14:0] addr, input [255:0] data);
        begin @(negedge clk); dma_write_en=1; dma_target=target; dma_addr=addr; dma_wdata=data; end
    endtask

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_U_Silu_FP.txt",     u_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Delta_FP.txt",      delta_mem);
        $readmemh("golden_all/block_00_layer00/Mam_B_Aligned_FP.txt",  B_mem);
        $readmemh("golden_all/block_00_layer00/Mam_C_Aligned_FP.txt",  C_mem);
        $readmemh("golden_all/block_00_layer00/Mam_A_signed.txt",      A_mem);
        $readmemh("golden_all/block_00_layer00/Mam_D_param.txt",       D_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Y_SSM_FP.txt",      exp_mem);

        rst=1; @(posedge clk); @(posedge clk); @(negedge clk); rst=0; @(posedge clk);

        // Load U → ram_a @ A_U_BASE
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = u_mem[(c_grp*16+lane)*T_TOT + t_cur];
                dma_wr(2'd0, A_U_BASE + t_cur*(D_INNER/16) + c_grp, word_tmp);
            end
        end

        // Load DELTA → ram_a @ A_DELTA_BASE
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = delta_mem[(c_grp*16+lane)*T_TOT + t_cur];
                dma_wr(2'd0, A_DELTA_BASE + t_cur*(D_INNER/16) + c_grp, word_tmp);
            end
        end

        // Load B → ram_a @ A_B_BASE (1 word per t, pack 16 state)
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            word_tmp=0;
            for (s=0; s<D_STATE; s=s+1)
                word_tmp[s*16+:16] = B_mem[s*T_TOT + t_cur];
            dma_wr(2'd0, A_B_BASE + t_cur, word_tmp);
        end

        // Load C → ram_a @ A_C_BASE
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            word_tmp=0;
            for (s=0; s<D_STATE; s=s+1)
                word_tmp[s*16+:16] = C_mem[s*T_TOT + t_cur];
            dma_wr(2'd0, A_C_BASE + t_cur, word_tmp);
        end

        // Load A → ram_w @ W_A_BASE (1 word per channel, pack 16 state)
        for (c=0; c<D_INNER; c=c+1) begin
            word_tmp=0;
            for (s=0; s<D_STATE; s=s+1)
                word_tmp[s*16+:16] = A_mem[c*D_STATE + s];
            dma_wr(2'd2, W_A_BASE + c, word_tmp);
        end

        // Load D → ram_const @ C_D_BASE (pack 16 ch/word)
        for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
            word_tmp=0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = D_mem[c_grp*16 + lane];
            dma_wr(2'd3, C_D_BASE + c_grp, word_tmp);
        end

        @(negedge clk); dma_write_en=0;

        @(negedge clk); start=1; @(negedge clk); start=0;
        $display("[FSM] M6 SSM scan running (T_TEST=%0d)...", T_TEST);
        wait (done_stage==1'b1);
        $display("[FSM] M6 done at %0t", $time);
        @(negedge clk); wait (done_stage==1'b0);

        // Readback Y_SSM from ram_b @ B_Y_SSM_BASE
        @(negedge clk); dma_read_en=1; dma_rtarget=2'd1;
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                @(negedge clk); dma_raddr = B_Y_SSM_BASE + t_cur*(D_INNER/16) + c_grp;
                @(posedge clk); @(negedge clk);
                readback = dma_rdata;
                for (lane=0; lane<16; lane=lane+1) begin
                    got_val = readback[lane*16+:16];
                    exp_val = exp_mem[(c_grp*16+lane)*T_TOT + t_cur];
                    compares = compares + 1;
                    if (got_val !== exp_val) begin
                        $display("FAIL t=%0d c=%0d got=%6d exp=%6d",
                            t_cur, c_grp*16+lane, got_val, exp_val);
                        errors = errors + 1;
                    end
                end
            end
        end
        dma_read_en=0;

        $display("");
        $display("---- tb_Mamba_Top_M6 summary: compares=%0d errors=%0d ----", compares, errors);
        if (errors==0) $display("===== TB M6 BYTE-EXACT PASS =====");
        else           $display("===== TB M6 FAIL =====");
        $finish;
    end

    initial begin #500000000; $display("ERROR: timeout"); $finish; end
endmodule
