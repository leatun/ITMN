`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_M_Cluster_P1 — byte-exact integration test of M_Cluster (16-lane).
//
// Reuses block-0 P1 goldens (pure MAC reduction over d_in + bias). Same
// arithmetic pattern as Mamba M1A/M1B/M3/M4/M8 → a P1 pass certifies the
// cluster's MAC+ADD datapath for the entire MAC-class of Mamba stages.
//
//   Per timestep t and per c_out_group of 16 channels:
//     1. MAC chain over c_in = 0..63   (16 lanes in parallel along c_out)
//          clear_acc=1 on c_in=0, else 0
//     2. After 64 MAC cycles, single ADD cycle to add per-lane bias
//     3. Compare 16 outputs vs golden
//
// Block 0: d_in=D_IN=64, d_out=D_OUT=64, T=T_TOT=1000.
// → c_out_grp count = D_OUT / 16 = 4 groups
//
// Test scope: first T_TEST timesteps × all 64 output channels.
// Set T_TEST small (2) for quick smoke; raise for full sweep.
//
// Files expected (in xsim working directory):
//   P1_Input_X.txt           (d_in=64, T=1000), row-major c-fast
//   P1_Weight_Fused.txt      (d_out=64, d_in=64), row-major
//   P1_Bias_Fused.txt        (d_out=64)
//   P1_Output_Golden_FP.txt  (d_out=64, T=1000), row-major
//
// Copy from: ITMN_Pytorch/golden_all/block_00_layer00/
// ============================================================================

module tb_M_Cluster_P1;

    // Dimensions — centralised in _parameter.v
    localparam D_IN   = `B0_D_MODEL;
    localparam D_OUT  = `B0_D_MODEL;
    localparam T_TOT  = `B0_T_TOT;
    localparam T_TEST = 1000;     // small smoke test; bump to 1000 for full

    // Clock / reset
    reg clk = 0;
    reg rst = 1;
    always #5 clk = ~clk;       // 100 MHz

    // M_Cluster ports (256-bit vectors = 16 lane × 16-bit)
    reg  [2:0]                 op_mode;
    reg                        clear_acc;
    reg  [16*`DATA_W-1:0]      in_W1_vec;
    reg  [16*`DATA_W-1:0]      in_H_ext;
    reg  [16*`DATA_W-1:0]      in_W2_vec     = 256'b0;
    reg  [16*`DATA_W-1:0]      in_X_vec      = 256'b0;
    reg                        h_from_rf     = 1'b0;
    reg  [8:0]                 h_rd_addr     = 9'b0;
    reg                        h_wr_en       = 1'b0;
    reg  [8:0]                 h_wr_addr     = 9'b0;
    reg                        h_wr_from_pe  = 1'b0;
    reg  [16*`DATA_W-1:0]      h_wr_data_ext = 256'b0;

    wire [16*`DATA_W-1:0]      out_vec;
    wire [`DATA_W+4:0]         y_reduce_out;

    M_Cluster #(
        .H_ADDR_W(9),
        .H_DEPTH (256)
    ) dut (
        .clk          (clk),
        .rst          (rst),
        .op_mode      (op_mode),
        .clear_acc    (clear_acc),
        .in_W1_vec    (in_W1_vec),
        .in_H_ext     (in_H_ext),
        .in_W2_vec    (in_W2_vec),
        .in_X_vec     (in_X_vec),
        .h_from_rf    (h_from_rf),
        .h_rd_addr    (h_rd_addr),
        .h_wr_en       (h_wr_en),
        .h_wr_addr    (h_wr_addr),
        .h_wr_from_pe (h_wr_from_pe),
        .h_wr_data_ext(h_wr_data_ext),
        .out_vec      (out_vec),
        .y_reduce_out (y_reduce_out)
    );

    // Golden storage
    reg [15:0] input_mem  [0:D_IN*T_TOT-1];
    reg [15:0] weight_mem [0:D_OUT*D_IN-1];
    reg [15:0] bias_mem   [0:D_OUT-1];
    reg [15:0] expect_mem [0:D_OUT*T_TOT-1];

    integer errors    = 0;
    integer checks    = 0;
    integer compares  = 0;
    integer t_cur, c_out_grp, c_in, lane;
    reg signed [15:0] got_val, expected_val;

    initial begin
        $readmemh("golden_all/block_00_layer00/P1_Input_X.txt",          input_mem);
        $readmemh("golden_all/block_00_layer00/P1_Weight_Fused.txt",     weight_mem);
        $readmemh("golden_all/block_00_layer00/P1_Bias_Fused.txt",       bias_mem);
        $readmemh("golden_all/block_00_layer00/P1_Output_Golden_FP.txt", expect_mem);

        // Reset
        op_mode   = `MAMBA_PE_IDLE;
        clear_acc = 0;
        in_W1_vec = 0;
        in_H_ext  = 0;
        rst = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // Main test loop — 16 lanes per group → D_OUT/16 = 4 groups for d_out=64
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1) begin
            for (c_out_grp = 0; c_out_grp < (D_OUT/16); c_out_grp = c_out_grp + 1) begin

                // --- MAC chain over c_in = 0..D_IN-1 ---
                for (c_in = 0; c_in < D_IN; c_in = c_in + 1) begin
                    @(negedge clk);
                    op_mode   = `MAMBA_PE_MAC;
                    clear_acc = (c_in == 0);
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        in_W1_vec[lane*16 +: 16] =
                            weight_mem[(c_out_grp*16 + lane)*D_IN + c_in];
                        in_H_ext [lane*16 +: 16] =
                            input_mem [c_in*T_TOT + t_cur];
                    end
                end

                // Wait for the last MAC posedge to latch into out_vec
                @(posedge clk);

                // --- ADD bias on top of MAC result ---
                @(negedge clk);
                op_mode   = `MAMBA_PE_ADD;
                clear_acc = 0;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    in_W1_vec[lane*16 +: 16] = out_vec[lane*16 +: 16];     // MAC final
                    in_H_ext [lane*16 +: 16] = bias_mem[c_out_grp*16 + lane];
                end

                // Wait for ADD result
                @(posedge clk);
                @(negedge clk);   // settle for sampling

                // Compare 16 lane outputs
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    expected_val = expect_mem[(c_out_grp*16 + lane)*T_TOT + t_cur];
                    got_val      = out_vec[lane*16 +: 16];
                    compares = compares + 1;
                    if (got_val !== expected_val) begin
                        $display("FAIL  t=%0d c_out=%0d  got=%6d (0x%04h)  exp=%6d (0x%04h)",
                                 t_cur, c_out_grp*16 + lane,
                                 got_val,      got_val      & 16'hFFFF,
                                 expected_val, expected_val & 16'hFFFF);
                        errors = errors + 1;
                    end
                end

                // Return to IDLE between groups
                @(negedge clk);
                op_mode = `MAMBA_PE_IDLE;
            end
        end

        $display("");
        $display("---- tb_M_Cluster_P1 summary ----");
        $display("  timesteps tested : %0d / %0d", T_TEST, T_TOT);
        $display("  total compares   : %0d", compares);
        $display("  errors           : %0d", errors);
        if (errors == 0)
            $display("===== TB P1 BYTE-EXACT PASS =====");
        else
            $display("===== TB P1 FAIL =====");
        $finish;
    end

    // Watchdog
    initial begin
        #5000000;
        $display("ERROR: simulation timeout");
        $finish;
    end

endmodule
