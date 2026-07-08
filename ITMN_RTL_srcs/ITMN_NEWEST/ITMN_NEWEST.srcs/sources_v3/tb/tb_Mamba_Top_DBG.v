`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_Top_DBG — 1-timestep debug run.
//   Chạy T_MAX=1 → intermediate RAM regions còn nguyên state của t=0
//   → dump từng stage output và so với goldens tương ứng.
//
// Có thể dump được sau 1 timestep (do aliasing):
//   Slot CIRC  (64..71): X_INNER[t=0]   (M1A output)
//   Slot 3     (48..55): Z_GATE[t=0]    (M1B output)
//   Slot 2     (32..39): DELTA[t=0]     (M5 output)
//   Slot 1     (16..23): Y_SSM[t=0]     (M6 output — cuối cùng ghi slot 1)
//   Slot 0     ( 0.. 7): Y_GATED[t=0]   (M7 output — cuối cùng ghi slot 0)
//   Slot BULK  (128..131): MAMBA_OUT[t=0] (M8 output)
//
// Không dump được (overwrite): X_NORM, X_CONV, U, X_PROJ.
// Nhưng nếu tất cả downstream match golden → upstream cũng correct.
// ============================================================================

module tb_Mamba_Top_DBG;

    localparam D_MODEL   = `B0_D_MODEL;   // 64
    localparam D_INNER   = `B0_D_INNER;   // 128
    localparam D_STATE   = `B0_D_STATE;   // 16
    localparam DT_RANK_V = `B0_DT_RANK;   // 4
    localparam N_PAD     = `B0_N_PAD;     // 48
    localparam T_TOT     = `B0_T_TOT;     // 1000
    localparam T_TEST    = 1;             // ONE timestep only

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
    localparam PT_X_INNER_CIRC = `PT_X_INNER_CIRC;
    localparam PT_Z_GATE       = `PT_Z_GATE;
    localparam PT_DELTA        = `PT_DELTA;
    localparam PT_Y_SSM        = `PT_Y_SSM;
    localparam PT_Y_GATED      = `PT_Y_GATED;

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
    reg [15:0] wo_mem   [0:D_MODEL*D_INNER-1];
    reg [15:0] wdw_mem  [0:D_INNER*4-1];
    reg [15:0] wxp_mem  [0:N_PAD*D_INNER-1];
    reg [15:0] wdt_mem  [0:D_INNER*DT_RANK_V-1];
    reg [15:0] wA_mem   [0:D_INNER*D_STATE-1];
    reg [15:0] gam_mem  [0:D_MODEL-1];
    reg [15:0] bdw_mem  [0:D_INNER-1];
    reg [15:0] bdt_mem  [0:D_INNER-1];
    reg [15:0] Dp_mem   [0:D_INNER-1];
    reg [15:0] xn_mem   [0:D_MODEL*T_TOT-1];

    // ----- Expected intermediates -----
    reg [15:0] xi_exp    [0:D_INNER*T_TOT-1];   // X_INNER (M1A)
    reg [15:0] zg_exp    [0:D_INNER*T_TOT-1];   // Z_GATE (M1B)
    reg [15:0] dlt_exp   [0:D_INNER*T_TOT-1];   // DELTA (M5)
    reg [15:0] yssm_exp  [0:D_INNER*T_TOT-1];   // Y_SSM (M6)
    reg [15:0] ygat_exp  [0:D_INNER*T_TOT-1];   // Y_GATED (M7)
    reg [15:0] mout_exp  [0:D_MODEL*T_TOT-1];   // MAMBA_OUT (M8)

    integer errors_xi = 0, errors_zg = 0, errors_dlt = 0;
    integer errors_yssm = 0, errors_ygat = 0, errors_mout = 0;
    integer c_out_grp, c_grp_out, c_in, k, t_cur, lane, c_grp_in;
    reg [255:0] word_tmp, readback;
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

    // ----- Compare 1 packed word vs golden (row-major: row=channel, col=t) -----
    task compare_word;
        input [15*8:0] name;
        input [14:0]   addr;
        input          use_dinner;   // 1 if d_inner channels, 0 if d_model
        input integer  c_grp;
        input integer  t_target;
        // Which golden array: 0=xi, 1=zg, 2=dlt, 3=yssm, 4=ygat, 5=mout
        input [2:0]    which;
        output integer errs;
        integer        lane_i, ch, idx;
        reg signed [15:0] g, e;
        integer        depth;
        begin
            errs = 0;
            depth = use_dinner ? D_INNER : D_MODEL;
            @(negedge clk);
            dma_read_en = 1; dma_rtarget = 2'd0;
            dma_raddr = addr;
            @(posedge clk); @(negedge clk);
            readback = dma_rdata;
            dma_read_en = 0;
            for (lane_i = 0; lane_i < 16; lane_i = lane_i + 1) begin
                ch  = c_grp * 16 + lane_i;
                idx = ch * T_TOT + t_target;
                g = readback[lane_i*16 +: 16];
                case (which)
                    3'd0: e = xi_exp[idx];
                    3'd1: e = zg_exp[idx];
                    3'd2: e = dlt_exp[idx];
                    3'd3: e = yssm_exp[idx];
                    3'd4: e = ygat_exp[idx];
                    3'd5: e = mout_exp[idx];
                    default: e = 16'sd0;
                endcase
                if (g !== e) begin
                    if (errs < 4)
                        $display("  %0s FAIL c=%0d got=%6d exp=%6d",
                                 name, ch, g, e);
                    errs = errs + 1;
                end
            end
        end
    endtask

    integer sub_err;

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

        $readmemh("golden_all/block_00_layer00/Mam_X_Inner_FP.txt",     xi_exp);
        $readmemh("golden_all/block_00_layer00/Mam_Z_Gate_FP.txt",      zg_exp);
        $readmemh("golden_all/block_00_layer00/Mam_Delta_FP.txt",       dlt_exp);
        $readmemh("golden_all/block_00_layer00/Mam_Y_SSM_FP.txt",       yssm_exp);
        $readmemh("golden_all/block_00_layer00/Mam_Y_Gated_FP.txt",     ygat_exp);
        $readmemh("golden_all/block_00_layer00/Mam_OutProj_FP.txt",     mout_exp);

        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // ---- DMA load all weights + consts + input (as tb_Mamba_Top_FULL) ----
        $display("[DMA] W_InProj_X");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_MODEL; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wx_mem[(c_out_grp*16+lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_X_BASE + c_out_grp*D_MODEL + c_in, word_tmp);
            end

        $display("[DMA] W_InProj_Z");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_MODEL; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wz_mem[(c_out_grp*16+lane)*D_MODEL + c_in];
                dma_wr(2'd2, W_INPROJ_Z_BASE + c_out_grp*D_MODEL + c_in, word_tmp);
            end

        $display("[DMA] W_OutProj");
        for (c_out_grp=0; c_out_grp<(D_MODEL/16); c_out_grp=c_out_grp+1)
            for (c_in=0; c_in<D_INNER; c_in=c_in+1) begin
                word_tmp = 256'b0;
                for (lane=0; lane<16; lane=lane+1)
                    word_tmp[lane*16+:16] = wo_mem[(c_out_grp*16+lane)*D_INNER + c_in];
                dma_wr(2'd2, W_OUTPROJ_BASE + c_out_grp*D_INNER + c_in, word_tmp);
            end

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
        $display("[FSM] pipeline running (T=1) ...");
        wait (done_all == 1'b1);
        $display("[FSM] done at %0t\n", $time);
        @(negedge clk); @(negedge clk);

        // ================================================================
        //  Dump từng stage, so với golden, dừng ở stage đầu tiên fail
        // ================================================================

        // ---- M1A: X_INNER at Slot CIRC (t%4=0, addr 64..71) ----
        $display("[CHECK] X_INNER (M1A) — read from CIRC slot 0");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1) begin
            compare_word("X_INNER", PT_X_INNER_CIRC + c_out_grp,
                         1, c_out_grp, 0, 3'd0, sub_err);
            errors_xi = errors_xi + sub_err;
        end
        $display("  X_INNER errors: %0d / %0d\n", errors_xi, D_INNER);

        // ---- M1B: Z_GATE at Slot 3 (addr 48..55) ----
        $display("[CHECK] Z_GATE (M1B) — read from Slot 3");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1) begin
            compare_word("Z_GATE ", PT_Z_GATE + c_out_grp,
                         1, c_out_grp, 0, 3'd1, sub_err);
            errors_zg = errors_zg + sub_err;
        end
        $display("  Z_GATE  errors: %0d / %0d\n", errors_zg, D_INNER);

        // ---- M5: DELTA at Slot 2 (addr 32..39) ----
        $display("[CHECK] DELTA (M5) — read from Slot 2");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1) begin
            compare_word("DELTA  ", PT_DELTA + c_out_grp,
                         1, c_out_grp, 0, 3'd2, sub_err);
            errors_dlt = errors_dlt + sub_err;
        end
        $display("  DELTA   errors: %0d / %0d\n", errors_dlt, D_INNER);

        // ---- M6: Y_SSM at Slot 1 (addr 16..23) ----
        $display("[CHECK] Y_SSM (M6) — read from Slot 1");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1) begin
            compare_word("Y_SSM  ", PT_Y_SSM + c_out_grp,
                         1, c_out_grp, 0, 3'd3, sub_err);
            errors_yssm = errors_yssm + sub_err;
        end
        $display("  Y_SSM   errors: %0d / %0d\n", errors_yssm, D_INNER);

        // ---- M7: Y_GATED at Slot 0 (addr 0..7) ----
        $display("[CHECK] Y_GATED (M7) — read from Slot 0");
        for (c_out_grp=0; c_out_grp<(D_INNER/16); c_out_grp=c_out_grp+1) begin
            compare_word("Y_GATED", PT_Y_GATED + c_out_grp,
                         1, c_out_grp, 0, 3'd4, sub_err);
            errors_ygat = errors_ygat + sub_err;
        end
        $display("  Y_GATED errors: %0d / %0d\n", errors_ygat, D_INNER);

        // ---- M8: MAMBA_OUT at Slot BULK (addr 128..131) ----
        $display("[CHECK] MAMBA_OUT (M8) — read from Slot BULK");
        for (c_out_grp=0; c_out_grp<(D_MODEL/16); c_out_grp=c_out_grp+1) begin
            compare_word("M_OUT  ", PT_MAMBA_OUT + c_out_grp,
                         0, c_out_grp, 0, 3'd5, sub_err);
            errors_mout = errors_mout + sub_err;
        end
        $display("  MAMBA_OUT errors: %0d / %0d\n", errors_mout, D_MODEL);

        // ================================================================
        //  Summary
        // ================================================================
        $display("---- tb_Mamba_Top_DBG summary (t=0) ----");
        $display("  M1A X_INNER : %0d errors", errors_xi);
        $display("  M1B Z_GATE  : %0d errors", errors_zg);
        $display("  M5  DELTA   : %0d errors", errors_dlt);
        $display("  M6  Y_SSM   : %0d errors", errors_yssm);
        $display("  M7  Y_GATED : %0d errors", errors_ygat);
        $display("  M8  MAMBA_OUT: %0d errors", errors_mout);
        $display("");
        $display("  First-failing stage → root cause suspect.");
        $finish;
    end

    initial begin
        #200000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
