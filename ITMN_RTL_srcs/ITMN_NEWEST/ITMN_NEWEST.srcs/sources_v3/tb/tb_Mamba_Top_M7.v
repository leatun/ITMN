`timescale 1ns/1ps
`include "_parameter.v"

// tb_Mamba_Top_M7 — byte-exact M7 gating: y_gated = SiLU(z_gate) * y_ssm.
//
// Block 0: T=1000, d_inner=128.
//   Inputs (DMA-loaded):
//     Z_GATE → ram_b @ B_Z_GATE_BASE=8000   (d_inner × T words, pack 16 ch / word)
//     Y_SSM  → ram_b @ B_Y_SSM_BASE=0       (d_inner × T words, pack 16 ch / word)
//   Output:
//     Y_GATED → ram_a @ A_Y_GATED_BASE=16000 (d_inner × T words)
//
// Goldens:
//   Mam_Z_Gate_FP.txt    (d_inner, T) — M1B output
//   Mam_Y_SSM_FP.txt     (d_inner, T) — M6 output
//   Mam_Y_Gated_FP.txt   (d_inner, T) — expected
//
// FSM per (t, c_grp): Z_PREF, Z_WAIT, Z_LATCH (silu cap), Y_PREF, Y_WAIT,
//                     MUL (cluster MAMBA_PE_MUL), LATCH, WRITE, NEXT = 9 cycles.
// T_TEST=4, c_grp=8 → 4*8*9 = 288 cycles. ~3 μs sim.

module tb_Mamba_Top_M7;
    localparam D_INNER = `B0_D_INNER;
    localparam T_TOT   = `B0_T_TOT;
    localparam T_TEST  = 1000;
    localparam B_Z_GATE_BASE  = `B_Z_GATE_BASE;
    localparam B_Y_SSM_BASE   = `B_Y_SSM_BASE;
    localparam A_Y_GATED_BASE = `A_Y_GATED_BASE;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done_stage, done_all;
    reg [3:0] run_stage = 4'd7;
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

    reg [15:0] z_mem    [0:D_INNER*T_TOT-1];
    reg [15:0] y_mem    [0:D_INNER*T_TOT-1];
    reg [15:0] exp_mem  [0:D_INNER*T_TOT-1];

    integer errors=0, compares=0;
    integer c_grp, t_cur, lane;
    reg [255:0] word_tmp, readback;
    reg signed [15:0] got_val, exp_val;

    task dma_wr(input [1:0] target, input [14:0] addr, input [255:0] data);
        begin @(negedge clk); dma_write_en=1; dma_target=target; dma_addr=addr; dma_wdata=data; end
    endtask

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_Z_Gate_FP.txt",  z_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Y_SSM_FP.txt",   y_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Y_Gated_FP.txt", exp_mem);

        rst=1; @(posedge clk); @(posedge clk); @(negedge clk); rst=0; @(posedge clk);

        // Load Z_GATE → ram_b @ B_Z_GATE_BASE
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = z_mem[(c_grp*16+lane)*T_TOT + t_cur];
                dma_wr(2'd1, B_Z_GATE_BASE + t_cur*(D_INNER/16) + c_grp, word_tmp);
            end
        end

        // Load Y_SSM → ram_b @ B_Y_SSM_BASE
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = y_mem[(c_grp*16+lane)*T_TOT + t_cur];
                dma_wr(2'd1, B_Y_SSM_BASE + t_cur*(D_INNER/16) + c_grp, word_tmp);
            end
        end

        @(negedge clk); dma_write_en=0;

        @(negedge clk); start=1; @(negedge clk); start=0;
        $display("[FSM] M7 gating running (T_TEST=%0d)...", T_TEST);
        wait (done_stage==1'b1);
        $display("[FSM] M7 done at %0t", $time);
        @(negedge clk); wait (done_stage==1'b0);

        // Readback Y_GATED from ram_a @ A_Y_GATED_BASE
        @(negedge clk); dma_read_en=1; dma_rtarget=2'd0;
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                @(negedge clk); dma_raddr = A_Y_GATED_BASE + t_cur*(D_INNER/16) + c_grp;
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
        $display("---- tb_Mamba_Top_M7 summary: compares=%0d errors=%0d ----", compares, errors);
        if (errors==0) $display("===== TB M7 BYTE-EXACT PASS =====");
        else           $display("===== TB M7 FAIL =====");
        $finish;
    end

    initial begin #10000000; $display("ERROR: timeout"); $finish; end
endmodule
