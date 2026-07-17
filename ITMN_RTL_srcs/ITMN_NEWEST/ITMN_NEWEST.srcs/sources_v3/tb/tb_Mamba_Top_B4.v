`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_B4 — cycle-count characterization of Mamba_Top on ITMN block 4.
//
// Block 4 dims (from _block_params.v / itmn_pipeline.py):
//   T=250, d_model=128, d_inner=256, d_state=16, dt_rank=8, n_pad=48.
//   → CH_OUT=8, CH_M=0 (encodes 16), DT_RANK=8, T_MAX=250.
//
// Post-refactor (2026-07-13): all weights preloaded permanently (streaming
// removed, ram_weight depth expanded to 8192, TDP enables MAC2 2R/cyc always).
// W_INPROJ_X, W_INPROJ_Z, W_OUTPROJ each occupy their own permanent slot.
//
// No byte-exact compare — only cycle-count reported (start → done_all).
// ============================================================================

module tb_Mamba_Top_B4;

    // ---- B4 dims (hard-coded, not from _parameter.v since only B0 defined there) ----
    localparam D_MODEL   = 128;
    localparam D_INNER   = 256;
    localparam D_STATE   = 16;
    localparam DT_RANK_V = 8;
    localparam N_PAD     = 48;
    localparam T_TOT     = 250;
    localparam T_TEST    = 250;

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
    reg  [3:0]  CH_OUT    = 4'd8;   // B4: d_model=128 → CH_OUT=8
    reg  [3:0]  CH_M      = 4'd0;   // B4: d_inner=256 → CH_M=16 (encoded as 0)
    reg  [3:0]  DT_RANK   = 4'd8;   // B4: dt_rank=8

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

    // ---- Golden storage (B4-sized) ----
    reg [15:0] wx_mem   [0:D_INNER*D_MODEL-1];      // 256*128 = 32768
    reg [15:0] wz_mem   [0:D_INNER*D_MODEL-1];
    reg [15:0] wo_mem   [0:D_MODEL*D_INNER-1];      // 128*256 = 32768
    reg [15:0] wdw_mem  [0:D_INNER*4-1];            // 256*4 = 1024
    reg [15:0] wxp_mem  [0:N_PAD*D_INNER-1];        // 48*256 = 12288
    reg [15:0] wdt_mem  [0:D_INNER*DT_RANK_V-1];    // 256*8 = 2048
    reg [15:0] wA_mem   [0:D_INNER*D_STATE-1];      // 256*16 = 4096
    reg [15:0] gam_mem  [0:D_MODEL-1];              // 128
    reg [15:0] bdw_mem  [0:D_INNER-1];              // 256
    reg [15:0] bdt_mem  [0:D_INNER-1];              // 256
    reg [15:0] Dp_mem   [0:D_INNER-1];              // 256
    reg [15:0] xn_mem   [0:D_MODEL*T_TOT-1];        // 128*250 = 32000

    integer c_out_grp, c_grp_out, c_in, k, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp;

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
        $readmemh("golden_all/block_04_layer06/Mam_W_InProj_X.txt",     wx_mem);
        $readmemh("golden_all/block_04_layer06/Mam_W_InProj_Z.txt",     wz_mem);
        $readmemh("golden_all/block_04_layer06/Mam_W_OutProj.txt",      wo_mem);
        $readmemh("golden_all/block_04_layer06/Mam_W_Conv.txt",         wdw_mem);
        $readmemh("golden_all/block_04_layer06/Mam_W_XProj.txt",        wxp_mem);
        $readmemh("golden_all/block_04_layer06/Mam_W_DtProj.txt",       wdt_mem);
        $readmemh("golden_all/block_04_layer06/Mam_A_signed.txt",       wA_mem);
        $readmemh("golden_all/block_04_layer06/Mam_W_Norm.txt",         gam_mem);
        $readmemh("golden_all/block_04_layer06/Mam_B_Conv.txt",         bdw_mem);
        $readmemh("golden_all/block_04_layer06/Mam_B_DtProj.txt",       bdt_mem);
        $readmemh("golden_all/block_04_layer06/Mam_D_param.txt",        Dp_mem);
        $readmemh("golden_all/block_04_layer06/P1_Output_Golden_FP.txt", xn_mem);

        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // ---- Preload W_InProj_X → SLOT_A ----
        $display("[DMA] W_InProj_X (%0d words)", (D_INNER/16)*D_MODEL);
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_MODEL; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wx_mem[(c_out_grp*16+lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_X_BASE + c_out_grp*D_MODEL + c_in, word_tmp);
            end

        // ---- Preload W_InProj_Z → SLOT_B ----
        $display("[DMA] W_InProj_Z");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_MODEL; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wz_mem[(c_out_grp*16+lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_Z_BASE + c_out_grp*D_MODEL + c_in, word_tmp);
            end

        // ---- Preload W_OutProj → dedicated permanent slot ----
        $display("[DMA] W_OutProj");
        for (c_out_grp=0; c_out_grp<(D_MODEL/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wo_mem[(c_out_grp*16+lane)*D_INNER + c_in];
                dma_wr(2'd2, W_OUTPROJ_BASE + c_out_grp*D_INNER + c_in, word_tmp);
            end

        // ---- Preload SMALLS (resident, per-block constants) ----
        $display("[DMA] W_DW");
        for (c_grp_in=0; c_grp_in<(D_INNER/16); c_grp_in=c_grp_in+1)
            for (k=0; k<4; k=k+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wdw_mem[(c_grp_in*16+lane)*4 + k];
                dma_wr(2'd2, W_DW_BASE + c_grp_in*4 + k, word_tmp);
            end

        $display("[DMA] W_XProj");
        for (c_grp_out=0; c_grp_out<(N_PAD/16); c_grp_out=c_grp_out+1)
            for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wxp_mem[(c_grp_out*16+lane)*D_INNER + c_in];
                dma_wr(2'd2, W_XPROJ_BASE + c_grp_out*D_INNER + c_in, word_tmp);
            end

        $display("[DMA] W_DtProj");
        for (c_grp_out=0; c_grp_out<(D_INNER/16); c_grp_out=c_grp_out+1)
            for (k=0; k<DT_RANK_V; k=k+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wdt_mem[(c_grp_out*16+lane)*DT_RANK_V + k];
                dma_wr(2'd2, W_DTPROJ_BASE + c_grp_out*DT_RANK_V + k, word_tmp);
            end

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

        // ---- INPUT (P1 output = block-4 RMSNorm input) ----
        $display("[DMA] INPUT (%0d words)", T_TEST*(D_MODEL/16));
        for (t_cur=0; t_cur<T_TEST; t_cur=t_cur+1)
            for (c_grp_in=0; c_grp_in<(D_MODEL/16); c_grp_in=c_grp_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = xn_mem[(c_grp_in*16+lane)*T_TOT + t_cur];
                dma_wr(2'd0, PT_INPUT + t_cur*(D_MODEL/16) + c_grp_in, word_tmp);
            end
        @(negedge clk); dma_write_en = 0;

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc_start = $time / 10;
        $display("[FSM] pipeline running (T=%0d), cyc_start=%0d", T_TEST, cyc_start);

        wait (done_all == 1'b1);
        cyc_end = $time / 10;

        @(negedge clk); @(negedge clk);

        $display("");
        $display("---- tb_Mamba_Top_B4 summary ----");
        $display("  T=%0d  cyc_start=%0d  cyc_end=%0d  total=%0d cycles",
                 T_TEST, cyc_start, cyc_end, cyc_end - cyc_start);
        $display("  cycles per timestep = %0d", (cyc_end - cyc_start) / T_TEST);
        $display("===== TB B4 CYCLE COUNT DONE =====");
        $finish;
    end

    initial begin
        #2000000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
