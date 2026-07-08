`timescale 1ns/1ps
`include "_parameter.v"

// tb_Mamba_Top_M2 — byte-exact M2 (depthwise conv 4-tap + bias, pre-SiLU).
//
// Block 0: T=1000, d_inner=128.
//   Input  X_INNER  in ram_b @ 0       — Mam_X_Inner_FP (d_inner, T)
//   Weight W_DW     in ram_w @ 1536    — Mam_W_Conv (d_inner, 4)
//   Bias   B_DW     in ram_const @ 8   — Mam_B_Conv (d_inner)
//   Output X_CONV   in ram_a @ 4000    — compare Mam_X_Conv_FP (d_inner, T)
//
// Packing:
//   X_INNER word @ (t * d_inner/16 + c_grp) = pack x_inner[c_grp*16+0..+15, t]
//   W_DW    word @ (c_grp * 4 + k)          = pack W_dw[c_grp*16+0..+15, k]
//   B_DW    word @ c_grp                    = pack B_dw[c_grp*16+0..+15]
//   X_CONV  word @ (t * d_inner/16 + c_grp) = pack x_conv[c_grp*16+0..+15, t]

module tb_Mamba_Top_M2;
    localparam D_INNER = `B0_D_INNER;
    localparam T_TOT   = `B0_T_TOT;
    localparam T_TEST  = 1000;
    localparam W_DW_BASE     = `W_DW_BASE;
    localparam C_B_DW_BASE   = `C_B_DW_BASE;
    localparam B_X_INNER_BASE = `B_X_INNER_BASE;
    localparam A_X_CONV_BASE  = `A_X_CONV_BASE;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done_stage, done_all;
    reg [3:0] run_stage = 4'd2;
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

    reg [15:0] x_mem  [0:D_INNER*T_TOT-1];
    reg [15:0] w_mem  [0:D_INNER*4-1];
    reg [15:0] b_mem  [0:D_INNER-1];
    reg [15:0] exp_mem[0:D_INNER*T_TOT-1];

    integer errors=0, compares=0;
    integer c_grp, c_in, t_cur, lane, k;
    reg [255:0] word_tmp, readback;
    reg signed [15:0] got_val, exp_val;

    task dma_wr(input [1:0] target, input [14:0] addr, input [255:0] data);
        begin @(negedge clk); dma_write_en=1; dma_target=target; dma_addr=addr; dma_wdata=data; end
    endtask

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_X_Inner_FP.txt", x_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_Conv.txt", w_mem);
        $readmemh("golden_all/block_00_layer00/Mam_B_Conv.txt", b_mem);
        $readmemh("golden_all/block_00_layer00/Mam_X_Conv_FP.txt", exp_mem);

        rst=1; @(posedge clk); @(posedge clk); @(negedge clk); rst=0; @(posedge clk);

        // Load X_INNER → ram_b @ 0
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = x_mem[(c_grp*16+lane)*T_TOT + t_cur];
                dma_wr(2'd1, B_X_INNER_BASE + t_cur*(D_INNER/16) + c_grp, word_tmp);
            end
        end

        // Load W_DW → ram_w @ 1536: word per (c_grp, k) packs 16 channels for that tap
        for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
            for (k=0; k<4; k=k+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = w_mem[(c_grp*16+lane)*4 + k];
                dma_wr(2'd2, W_DW_BASE + c_grp*4 + k, word_tmp);
            end
        end

        // Load B_DW → ram_const @ 8
        for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
            word_tmp=0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = b_mem[c_grp*16 + lane];
            dma_wr(2'd3, C_B_DW_BASE + c_grp, word_tmp);
        end

        @(negedge clk); dma_write_en=0;

        @(negedge clk); start=1;
        @(negedge clk); start=0;
        $display("[FSM] M2 running...");
        wait (done_stage==1'b1);
        $display("[FSM] M2 done at %0t", $time);
        @(negedge clk); wait (done_stage==1'b0);

        // Readback ram_a @ A_X_CONV_BASE
        @(negedge clk); dma_read_en=1; dma_rtarget=2'd0;
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp=0; c_grp<(D_INNER/16); c_grp=c_grp+1) begin
                @(negedge clk); dma_raddr = A_X_CONV_BASE + t_cur*(D_INNER/16) + c_grp;
                @(posedge clk); @(negedge clk);
                readback = dma_rdata;
                for (lane=0; lane<16; lane=lane+1) begin
                    got_val = readback[lane*16+:16];
                    exp_val = exp_mem[(c_grp*16+lane)*T_TOT + t_cur];
                    compares = compares + 1;
                    if (got_val !== exp_val) begin
                        $display("FAIL t=%0d c=%0d got=%6d (0x%04h) exp=%6d (0x%04h)",
                            t_cur, c_grp*16+lane, got_val, got_val&16'hFFFF, exp_val, exp_val&16'hFFFF);
                        errors = errors + 1;
                    end
                end
            end
        end
        dma_read_en=0;

        $display("");
        $display("---- tb_Mamba_Top_M2 summary ----");
        $display("  compares: %0d  errors: %0d", compares, errors);
        if (errors==0) $display("===== TB M2 BYTE-EXACT PASS =====");
        else           $display("===== TB M2 FAIL =====");
        $finish;
    end

    initial begin #50000000; $display("ERROR: timeout"); $finish; end
endmodule
