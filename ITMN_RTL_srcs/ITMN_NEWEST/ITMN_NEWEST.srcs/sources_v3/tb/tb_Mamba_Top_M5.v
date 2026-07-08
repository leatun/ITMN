`timescale 1ns/1ps
`include "_parameter.v"

// tb_Mamba_Top_M5 — byte-exact M5 (dt_proj + bias + softplus).
//
// Block 0: T=1000, d_inner=128, dt_rank=4.
//   Input  X_PROJ  in ram_a @ 12000  — Mam_X_Proj_FP (n_pad=48, T)
//     FSM only uses first dt_rank=4 lanes of c_grp=0 word per t.
//   Weight W_DT    in ram_w @ 2368   — Mam_W_DtProj (d_inner, dt_rank) = (128, 4)
//   Bias   B_DT    in ram_const @ 24 — Mam_B_DtProj (d_inner) = 128 vals
//   Output DELTA   in ram_a @ 16000  — compare Mam_Delta_FP (d_inner, T)
//
// W_DT pack: word @ (c_grp * dt_rank + dt_idx) holds W[c_grp*16+0..+15, dt_idx]
// B_DT pack: word @ c_grp holds B[c_grp*16+0..+15]

module tb_Mamba_Top_M5;
    localparam D_INNER = `B0_D_INNER;
    localparam N_PAD   = `B0_N_PAD;
    localparam DT_RANK_VAL = `B0_DT_RANK;
    localparam T_TOT   = `B0_T_TOT;
    localparam T_TEST  = 600;
    localparam W_DTPROJ_BASE = `W_DTPROJ_BASE;
    localparam C_B_DT_BASE   = `C_B_DT_BASE;
    localparam A_X_PROJ_BASE = `A_X_PROJ_BASE;
    localparam A_DELTA_BASE  = `A_DELTA_BASE;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done_stage, done_all;
    reg [3:0] run_stage = 4'd5;
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

    reg [15:0] xp_mem   [0:N_PAD*T_TOT-1];
    reg [15:0] w_mem    [0:D_INNER*DT_RANK_VAL-1];
    reg [15:0] b_mem    [0:D_INNER-1];
    reg [15:0] exp_mem  [0:D_INNER*T_TOT-1];

    integer errors=0, compares=0;
    integer c_grp_out, c_in, t_cur, lane, c_grp_in, k;
    reg [255:0] word_tmp, readback;
    reg signed [15:0] got_val, exp_val;

    task dma_wr(input [1:0] target, input [14:0] addr, input [255:0] data);
        begin @(negedge clk); dma_write_en=1; dma_target=target; dma_addr=addr; dma_wdata=data; end
    endtask

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_X_Proj_FP.txt",  xp_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_DtProj.txt",   w_mem);
        $readmemh("golden_all/block_00_layer00/Mam_B_DtProj.txt",   b_mem);
        $readmemh("golden_all/block_00_layer00/Mam_Delta_FP.txt",   exp_mem);

        rst=1; @(posedge clk); @(posedge clk); @(negedge clk); rst=0; @(posedge clk);

        // Load X_PROJ → ram_a @ 12000 (need 3 c_grp per t)
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp_in=0; c_grp_in<(N_PAD/16); c_grp_in=c_grp_in+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = xp_mem[(c_grp_in*16+lane)*T_TOT + t_cur];
                dma_wr(2'd0, A_X_PROJ_BASE + t_cur*(N_PAD/16) + c_grp_in, word_tmp);
            end
        end

        // Load W_DT → ram_w @ 2368
        for (c_grp_out=0; c_grp_out<(D_INNER/16); c_grp_out=c_grp_out+1) begin
            for (k=0; k<DT_RANK_VAL; k=k+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = w_mem[(c_grp_out*16+lane)*DT_RANK_VAL + k];
                dma_wr(2'd2, W_DTPROJ_BASE + c_grp_out*DT_RANK_VAL + k, word_tmp);
            end
        end

        // Load B_DT → ram_const @ 24
        for (c_grp_out=0; c_grp_out<(D_INNER/16); c_grp_out=c_grp_out+1) begin
            word_tmp=0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = b_mem[c_grp_out*16 + lane];
            dma_wr(2'd3, C_B_DT_BASE + c_grp_out, word_tmp);
        end

        @(negedge clk); dma_write_en=0;

        @(negedge clk); start=1; @(negedge clk); start=0;
        $display("[FSM] M5 running...");
        wait (done_stage==1'b1);
        @(negedge clk); wait (done_stage==1'b0);

        // Readback ram_b @ A_DELTA_BASE (M5 bank_sel=0 → writes ram_b)
        @(negedge clk); dma_read_en=1; dma_rtarget=2'd1;
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp_out=0; c_grp_out<(D_INNER/16); c_grp_out=c_grp_out+1) begin
                @(negedge clk); dma_raddr = A_DELTA_BASE + t_cur*(D_INNER/16) + c_grp_out;
                @(posedge clk); @(negedge clk);
                readback = dma_rdata;
                for (lane=0; lane<16; lane=lane+1) begin
                    got_val = readback[lane*16+:16];
                    exp_val = exp_mem[(c_grp_out*16+lane)*T_TOT + t_cur];
                    compares = compares + 1;
                    if (got_val !== exp_val) begin
                        $display("FAIL t=%0d c=%0d got=%6d exp=%6d",
                            t_cur, c_grp_out*16+lane, got_val, exp_val);
                        errors = errors + 1;
                    end
                end
            end
        end
        dma_read_en=0;

        $display("");
        $display("---- tb_Mamba_Top_M5 summary: compares=%0d errors=%0d ----", compares, errors);
        if (errors==0) $display("===== TB M5 BYTE-EXACT PASS =====");
        else           $display("===== TB M5 FAIL =====");
        $finish;
    end

    initial begin #100000000; $display("ERROR: timeout"); $finish; end
endmodule
