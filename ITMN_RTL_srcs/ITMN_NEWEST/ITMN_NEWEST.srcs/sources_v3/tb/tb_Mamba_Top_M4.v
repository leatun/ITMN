`timescale 1ns/1ps
`include "_parameter.v"

// tb_Mamba_Top_M4 — byte-exact M4 (x_proj MAC reduction).
//
// Block 0: T=1000, d_inner=128, n_pad=48 (3 c_grp output).
//   Input  U      in ram_b @ 0       — Mam_U_Silu_FP    (d_inner, T)
//   Weight W_XP   in ram_w @ 1600    — Mam_W_XProj      (n_pad, d_inner) = (48, 128)
//   Output X_PROJ in ram_a @ 12000   — compare Mam_X_Proj_FP (n_pad, T)
//
// W_XP word @ (c_grp * d_inner + c_in) packs W[c_grp*16+0..+15, c_in].
// Output X_PROJ word @ (t * 3 + c_grp) packs xproj[c_grp*16+0..+15, t].
//
// Note: file has padded zeros for rows [n_act..n_pad-1].

module tb_Mamba_Top_M4;
    localparam D_INNER = `B0_D_INNER;
    localparam N_PAD   = `B0_N_PAD;
    localparam T_TOT   = `B0_T_TOT;
    localparam T_TEST  = 1000;
    localparam W_XPROJ_BASE = `W_XPROJ_BASE;
    localparam B_U_BASE     = `B_U_BASE;
    localparam A_X_PROJ_BASE= `A_X_PROJ_BASE;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg start = 0;
    wire done_stage, done_all;
    reg [3:0] run_stage = 4'd4;
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

    reg [15:0] u_mem   [0:D_INNER*T_TOT-1];
    reg [15:0] w_mem   [0:N_PAD*D_INNER-1];
    reg [15:0] exp_mem [0:N_PAD*T_TOT-1];

    integer errors=0, compares=0;
    integer c_grp_out, c_in, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp, readback;
    reg signed [15:0] got_val, exp_val;

    task dma_wr(input [1:0] target, input [14:0] addr, input [255:0] data);
        begin @(negedge clk); dma_write_en=1; dma_target=target; dma_addr=addr; dma_wdata=data; end
    endtask

    initial begin
        $readmemh("golden_all/block_00_layer00/Mam_U_Silu_FP.txt", u_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_XProj.txt", w_mem);
        $readmemh("golden_all/block_00_layer00/Mam_X_Proj_FP.txt", exp_mem);

        rst=1; @(posedge clk); @(posedge clk); @(negedge clk); rst=0; @(posedge clk);

        // Load U → ram_b @ 0
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp_in=0; c_grp_in<(D_INNER/16); c_grp_in=c_grp_in+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = u_mem[(c_grp_in*16+lane)*T_TOT + t_cur];
                dma_wr(2'd1, B_U_BASE + t_cur*(D_INNER/16) + c_grp_in, word_tmp);
            end
        end

        // Load W_XP → ram_w @ 1600
        for (c_grp_out=0; c_grp_out<(N_PAD/16); c_grp_out=c_grp_out+1) begin
            for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
                word_tmp=0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = w_mem[(c_grp_out*16+lane)*D_INNER + c_in];
                dma_wr(2'd2, W_XPROJ_BASE + c_grp_out*D_INNER + c_in, word_tmp);
            end
        end
        @(negedge clk); dma_write_en=0;

        @(negedge clk); start=1; @(negedge clk); start=0;
        $display("[FSM] M4 running...");
        wait (done_stage==1'b1);
        @(negedge clk); wait (done_stage==1'b0);

        // Readback ram_a @ A_X_PROJ_BASE
        @(negedge clk); dma_read_en=1; dma_rtarget=2'd0;
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_grp_out=0; c_grp_out<(N_PAD/16); c_grp_out=c_grp_out+1) begin
                @(negedge clk); dma_raddr = A_X_PROJ_BASE + t_cur*(N_PAD/16) + c_grp_out;
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
        $display("---- tb_Mamba_Top_M4 summary: compares=%0d errors=%0d ----", compares, errors);
        if (errors==0) $display("===== TB M4 BYTE-EXACT PASS =====");
        else           $display("===== TB M4 FAIL =====");
        $finish;
    end

    initial begin #100000000; $display("ERROR: timeout"); $finish; end
endmodule
