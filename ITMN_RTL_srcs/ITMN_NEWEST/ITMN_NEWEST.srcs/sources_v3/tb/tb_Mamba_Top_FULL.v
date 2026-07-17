`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_FULL — byte-exact test of per-timestep chained pipeline
//   (RMSNorm → M1A → M1B → M2 → M3 → M4 → M5 → M6 → M7 → M8) over T=1000.
//
// Block 0 dimensions: d_model=64, d_inner=128, d_state=16, dt_rank=4, T=1000.
//
// Flow:
//   1. DMA-load all weights + constants + INPUT (P1 output = RMSNorm input)
//   2. Pulse start
//   3. Wait done_all (H_INIT + full T loop through all 10 stages)
//   4. DMA-read MAMBA_OUT, compare with Mam_OutProj_FP
// ============================================================================

module tb_Mamba_Top_FULL;

    localparam D_MODEL   = `B0_D_MODEL;
    localparam D_INNER   = `B0_D_INNER;
    localparam D_STATE   = `B0_D_STATE;
    localparam DT_RANK_V = `B0_DT_RANK;
    localparam N_PAD     = `B0_N_PAD;
    localparam T_TOT     = `B0_T_TOT;
    localparam T_TEST    = 1000;

    localparam W_INPROJ_X_BASE = `W_INPROJ_X_BASE;
    localparam W_INPROJ_Z_BASE = `W_INPROJ_Z_BASE;
    localparam W_OUTPROJ_BASE  = `W_OUTPROJ_BASE;
    localparam W_DW_BASE       = `W_DW_BASE;
    localparam W_XPROJ_BASE    = `W_XPROJ_BASE;
    localparam W_DTPROJ_BASE   = `W_DTPROJ_BASE;
    localparam W_A_BASE        = `W_A_BASE;
    localparam C_W_NORM_BASE   = `C_W_NORM_BASE;
    localparam C_B_DW_BASE     = `C_B_DW_BASE;
    localparam C_B_DT_BASE     = `C_B_DT_BASE;
    localparam C_D_PARAM_BASE  = `C_D_PARAM_BASE;
    localparam PT_INPUT        = `PT_INPUT;
    localparam PT_MAMBA_OUT    = `PT_MAMBA_OUT;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg         start = 0;
    wire        done_stage, done_all;
    reg  [3:0]  run_stage = 4'd0;
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
        .clk(clk), .rst(rst), .start(start),
        .done_stage(done_stage), .done_all(done_all),
        .run_stage(run_stage), .T_MAX(T_MAX),
        .CH_OUT(CH_OUT), .CH_M(CH_M), .DT_RANK(DT_RANK),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // ----- Golden storage -----
    reg [15:0] wx_mem   [0:D_INNER*D_MODEL-1];
    reg [15:0] wz_mem   [0:D_INNER*D_MODEL-1];
    reg [15:0] wo_mem   [0:D_MODEL*D_INNER-1];       // W_OutProj (d_model=d_out, d_inner)
    reg [15:0] wdw_mem  [0:D_INNER*4-1];             // depthwise conv 4-tap
    reg [15:0] wxp_mem  [0:N_PAD*D_INNER-1];         // x_proj (n_pad, d_inner)
    reg [15:0] wdt_mem  [0:D_INNER*DT_RANK_V-1];     // dt_proj (d_inner, dt_rank)
    reg [15:0] wA_mem   [0:D_INNER*D_STATE-1];       // A signed
    reg [15:0] gam_mem  [0:D_MODEL-1];               // RMSNorm gamma
    reg [15:0] bdw_mem  [0:D_INNER-1];               // dwconv bias
    reg [15:0] bdt_mem  [0:D_INNER-1];               // dt bias
    reg [15:0] Dp_mem   [0:D_INNER-1];               // D param
    reg [15:0] xn_mem   [0:D_MODEL*T_TOT-1];         // P1 output = RMSNorm input (d_model, T)
    reg [15:0] exp_mem  [0:D_MODEL*T_TOT-1];         // MAMBA_OUT expected (d_out=d_model, T)

    integer errors = 0, compares = 0;
    integer c_out_grp, c_grp_out, c_in, k, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp, readback;
    reg signed [15:0] got_val, exp_val;

    integer cyc_start;
    integer cyc_end;
    initial begin
        cyc_start = 0;
        cyc_end   = 0;
    end

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
        $readmemh("golden_all/block_00_layer00/Mam_W_InProj_X.txt",     wx_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_InProj_Z.txt",     wz_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_OutProj.txt",      wo_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_Conv.txt",         wdw_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_XProj.txt",        wxp_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_DtProj.txt",       wdt_mem);
        $readmemh("golden_all/block_00_layer00/Mam_A_signed.txt",       wA_mem);
        $readmemh("golden_all/block_00_layer00/Mam_W_Norm.txt",         gam_mem);
        $readmemh("golden_all/block_00_layer00/Mam_B_Conv.txt",         bdw_mem);
        $readmemh("golden_all/block_00_layer00/Mam_B_DtProj.txt",       bdt_mem);
        $readmemh("golden_all/block_00_layer00/Mam_D_param.txt",        Dp_mem);
        $readmemh("golden_all/block_00_layer00/P1_Output_Golden_FP.txt", xn_mem);
        $readmemh("golden_all/block_00_layer00/Mam_OutProj_FP.txt",     exp_mem);

        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // ---- W_InProj_X → ram_weight ----
        $display("[DMA] W_InProj_X (%0d words)", (D_INNER/16)*D_MODEL);
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_MODEL; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wx_mem[(c_out_grp*16+lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_X_BASE + c_out_grp*D_MODEL + c_in, word_tmp);
            end

        // ---- W_InProj_Z ----
        $display("[DMA] W_InProj_Z");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_MODEL; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wz_mem[(c_out_grp*16+lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_Z_BASE + c_out_grp*D_MODEL + c_in, word_tmp);
            end

        // ---- W_OutProj (d_out=d_model, d_inner) ----
        $display("[DMA] W_OutProj");
        for (c_out_grp=0; c_out_grp<(D_MODEL/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wo_mem[(c_out_grp*16+lane)*D_INNER + c_in];
                dma_wr(2'd2, W_OUTPROJ_BASE + c_out_grp*D_INNER + c_in, word_tmp);
            end

        // ---- W_DW: 4-tap depthwise ----
        $display("[DMA] W_DW");
        for (c_grp_in=0; c_grp_in<(D_INNER/16); c_grp_in=c_grp_in+1)
            for (k=0; k<4; k=k+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wdw_mem[(c_grp_in*16+lane)*4 + k];
                dma_wr(2'd2, W_DW_BASE + c_grp_in*4 + k, word_tmp);
            end

        // ---- W_XProj (n_pad, d_inner) ----
        $display("[DMA] W_XProj");
        for (c_grp_out=0; c_grp_out<(N_PAD/16); c_grp_out=c_grp_out+1)
            for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wxp_mem[(c_grp_out*16+lane)*D_INNER + c_in];
                dma_wr(2'd2, W_XPROJ_BASE + c_grp_out*D_INNER + c_in, word_tmp);
            end

        // ---- W_DtProj (d_inner, dt_rank) ----
        $display("[DMA] W_DtProj");
        for (c_grp_out=0; c_grp_out<(D_INNER/16); c_grp_out=c_grp_out+1)
            for (k=0; k<DT_RANK_V; k=k+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wdt_mem[(c_grp_out*16+lane)*DT_RANK_V + k];
                dma_wr(2'd2, W_DTPROJ_BASE + c_grp_out*DT_RANK_V + k, word_tmp);
            end

        // ---- W_A signed (d_inner, d_state) — 1 word per channel ----
        $display("[DMA] W_A");
        for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
            word_tmp = 256'b0;
            for (lane=0; lane<D_STATE; lane=lane+1)
                word_tmp[lane*16+:16] = wA_mem[c_in*D_STATE + lane];
            dma_wr(2'd2, W_A_BASE + c_in, word_tmp);
        end

        // ---- Constants ----
        $display("[DMA] gamma");
        for (c_grp_in=0; c_grp_in<(D_MODEL/16); c_grp_in=c_grp_in+1) begin
            word_tmp = 256'b0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = gam_mem[c_grp_in*16 + lane];
            dma_wr(2'd3, C_W_NORM_BASE + c_grp_in, word_tmp);
        end

        $display("[DMA] B_DW");
        for (c_grp_in=0; c_grp_in<(D_INNER/16); c_grp_in=c_grp_in+1) begin
            word_tmp = 256'b0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = bdw_mem[c_grp_in*16 + lane];
            dma_wr(2'd3, C_B_DW_BASE + c_grp_in, word_tmp);
        end

        $display("[DMA] B_DT");
        for (c_grp_in=0; c_grp_in<(D_INNER/16); c_grp_in=c_grp_in+1) begin
            word_tmp = 256'b0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = bdt_mem[c_grp_in*16 + lane];
            dma_wr(2'd3, C_B_DT_BASE + c_grp_in, word_tmp);
        end

        $display("[DMA] D_param");
        for (c_grp_in=0; c_grp_in<(D_INNER/16); c_grp_in=c_grp_in+1) begin
            word_tmp = 256'b0;
            for (lane=0; lane<16; lane=lane+1)
                word_tmp[lane*16+:16] = Dp_mem[c_grp_in*16 + lane];
            dma_wr(2'd3, C_D_PARAM_BASE + c_grp_in, word_tmp);
        end

        // ---- INPUT (P1 output = block-0 RMSNorm input) ----
        //   Pack per (t, c_grp_in) → 16 channels/word
        //   addr = PT_INPUT + t * d_model/16 + c_grp_in
        $display("[DMA] INPUT (%0d words)", T_TEST*(D_MODEL/16));
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1)
            for (c_grp_in=0; c_grp_in<(D_MODEL/16); c_grp_in=c_grp_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = xn_mem[(c_grp_in*16+lane)*T_TOT + t_cur];
                dma_wr(2'd0, PT_INPUT + t_cur*(D_MODEL/16) + c_grp_in, word_tmp);
            end
        @(negedge clk); dma_write_en = 0;

        // ---- Start pipeline ----
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc_start = $time / 10;
        $display("[FSM] pipeline running (T=%0d), cyc_start=%0d", T_TEST, cyc_start);
        wait (done_all == 1'b1);
        cyc_end = $time / 10;
        $display("[FSM] done at %0t, cyc_end=%0d", $time, cyc_end);
        @(negedge clk); @(negedge clk);

        // ---- Read back MAMBA_OUT and compare ----
        @(negedge clk);
        dma_read_en = 1;
        dma_rtarget = 2'd0;
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1) begin
            for (c_out_grp=0; c_out_grp<(D_MODEL/16); c_out_grp=c_out_grp+1) begin
                @(negedge clk);
                dma_raddr = PT_MAMBA_OUT + t_cur*(D_MODEL/16) + c_out_grp;
                @(posedge clk); @(negedge clk);
                readback = dma_rdata;
                for (lane=0; lane<16; lane=lane+1) begin
                    got_val = readback[lane*16 +: 16];
                    exp_val = exp_mem[(c_out_grp*16+lane)*T_TOT + t_cur];
                    compares = compares + 1;
                    if (got_val !== exp_val) begin
                        if (errors < 20)
                            $display("FAIL t=%0d c_out=%0d got=%6d exp=%6d",
                                     t_cur, c_out_grp*16+lane, got_val, exp_val);
                        errors = errors + 1;
                    end
                end
            end
        end
        dma_read_en = 0;

        $display("");
        $display("---- tb_Mamba_Top_FULL summary ----");
        $display("  T=%0d compares=%0d errors=%0d", T_TEST, compares, errors);
        $display("  cyc_start=%0d  cyc_end=%0d  total=%0d cycles",
                 cyc_start, cyc_end, cyc_end - cyc_start);
        $display("  cycles per timestep = %0d", (cyc_end - cyc_start) / T_TEST);
        if (errors == 0) $display("===== TB FULL BYTE-EXACT PASS =====");
        else             $display("===== TB FULL FAIL =====");
        $finish;
    end

    initial begin
        #2000000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
