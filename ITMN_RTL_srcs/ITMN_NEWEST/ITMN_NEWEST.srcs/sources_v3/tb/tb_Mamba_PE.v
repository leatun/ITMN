`timescale 1ns/1ps
`include "_parameter.v"

// ============================================================================
// tb_Mamba_PE — direct-mode functional test for Mamba_PE.
//
// Test plan:
//   T1. IDLE         : out_val = 0
//   T2. MUL          : 1 product, sat16(>>FB)
//   T3. ADD          : simple int16 add with sat
//   T4. MAC chain    : clear_acc=1 first sample, then accumulate 3 samples
//   T5. SSM fused    : (W1*H + W2*X) >> FB in one cycle
//   T6. Sat boundary : positive overflow & negative overflow
//
// Run:
//   xsim → compile -sv tb_Mamba_PE.v Mamba_PE.v _parameter.v
//          elaborate -top tb_Mamba_PE
//          run all
//
// PASS if final "TB DONE — all checks passed" line appears with errors=0.
// ============================================================================

module tb_Mamba_PE;

    reg                      clk = 0;
    reg                      rst = 1;
    reg  [2:0]               op_mode = `MAMBA_PE_IDLE;
    reg                      clear_acc = 0;
    reg  signed [`DATA_W-1:0] in_W1 = 0;
    reg  signed [`DATA_W-1:0] in_H  = 0;
    reg  signed [`DATA_W-1:0] in_W2 = 0;
    reg  signed [`DATA_W-1:0] in_X  = 0;
    wire signed [`DATA_W-1:0] out_val;
    wire signed [`ACC_W-1:0]  acc_raw;

    integer errors = 0;

    Mamba_PE dut (
        .clk(clk), .rst(rst),
        .op_mode(op_mode), .clear_acc(clear_acc),
        .in_W1(in_W1), .in_H(in_H), .in_W2(in_W2), .in_X(in_X),
        .out_val(out_val), .acc_raw(acc_raw)
    );

    always #5 clk = ~clk;   // 100 MHz

    task check;
        input signed [`DATA_W-1:0] got;
        input signed [`DATA_W-1:0] exp;
        input [255:0]              label;
        begin
            if (got !== exp) begin
                $display("FAIL  %0s : got %0d (0x%04h), expected %0d (0x%04h)",
                         label, got, got & 16'hFFFF, exp, exp & 16'hFFFF);
                errors = errors + 1;
            end else begin
                $display("ok    %0s : %0d (0x%04h)", label, got, got & 16'hFFFF);
            end
        end
    endtask

    // Reference helper: sat16 of a signed 40-bit value
    function automatic signed [`DATA_W-1:0] ref_sat16;
        input signed [`ACC_W-1:0] v;
        begin
            if      (v >  40'sd32767)  ref_sat16 = 16'sh7FFF;
            else if (v < -40'sd32768)  ref_sat16 = 16'sh8000;
            else                        ref_sat16 = v[`DATA_W-1:0];
        end
    endfunction

    // Reference for MAC/MUL/SSM: shift then saturate
    function automatic signed [`DATA_W-1:0] ref_shift_sat;
        input signed [`ACC_W-1:0] v;
        begin
            ref_shift_sat = ref_sat16(v >>> `FRAC_BITS);
        end
    endfunction

    // Drive inputs in cycle N, sample out_val in cycle N+1.
    task drive;
        input [2:0]               m;
        input                     ce;
        input signed [`DATA_W-1:0] w1, h, w2, x;
        begin
            @(negedge clk);
            op_mode   = m;
            clear_acc = ce;
            in_W1     = w1;
            in_H      = h;
            in_W2     = w2;
            in_X      = x;
        end
    endtask

    initial begin
        // Reset
        rst = 1;
        @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;

        // T1. IDLE — out should stay 0 after reset
        drive(`MAMBA_PE_IDLE, 0, 16'sd0, 16'sd0, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sd0, "T1.IDLE out=0");

        // T2. MUL: 2.0 * 0.5 = 1.0 in Q4.11
        //     W1=4096 (2.0), H=1024 (0.5) → mul=4194304, >>11 = 2048 (=1.0)
        drive(`MAMBA_PE_MUL, 0, 16'sd4096, 16'sd1024, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sd2048, "T2.MUL 2.0*0.5=1.0");

        // T3. ADD: 100 + 200 = 300
        drive(`MAMBA_PE_ADD, 0, 16'sd100, 16'sd200, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sd300, "T3.ADD 100+200=300");

        // T4. MAC chain — accumulate 3 products with clear on first
        //     sample 1: clear=1, W1=4096, H=2048  → m1 = 4096*2048 = 8388608
        //                                            acc = 8388608, out = sat(8388608>>11)=4096
        //     sample 2: clear=0, W1=2048, H=2048  → m1 = 4194304
        //                                            acc = 8388608+4194304 = 12582912
        //                                            out = sat(12582912>>11) = 6144
        //     sample 3: clear=0, W1=1024, H=1024  → m1 = 1048576
        //                                            acc = 12582912+1048576 = 13631488
        //                                            out = sat(13631488>>11) = 6656
        drive(`MAMBA_PE_MAC, 1, 16'sd4096, 16'sd2048, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sd4096, "T4.MAC s1 (clear)");
        drive(`MAMBA_PE_MAC, 0, 16'sd2048, 16'sd2048, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sd6144, "T4.MAC s2 (accum)");
        drive(`MAMBA_PE_MAC, 0, 16'sd1024, 16'sd1024, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sd6656, "T4.MAC s3 (accum)");

        // T5. SSM fused: dA*h + dB*x
        //     W1=4096 (dA=2.0),  H=2048 (h=1.0)   → m1 = 8388608
        //     W2=1024 (dB=0.5),  X=4096 (x=2.0)   → m2 = 4194304
        //     sum = 12582912 → >>11 = 6144 (= 3.0)
        drive(`MAMBA_PE_SSM, 0, 16'sd4096, 16'sd2048, 16'sd1024, 16'sd4096);
        @(posedge clk); #1;
        check(out_val, 16'sd6144, "T5.SSM dA*h+dB*x=3.0");

        // T6. Saturation
        //     a) positive overflow MUL: 16'h7FFF * 16'h7FFF >> 11 → way > 32767
        drive(`MAMBA_PE_MUL, 0, 16'sh7FFF, 16'sh7FFF, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sh7FFF, "T6a.MUL pos overflow → +max");
        //     b) negative overflow MUL: 16'h7FFF * 16'h8000 >> 11 → way < -32768
        drive(`MAMBA_PE_MUL, 0, 16'sh7FFF, 16'sh8000, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sh8000, "T6b.MUL neg overflow → -max");
        //     c) ADD positive overflow: 32000 + 32000 = 64000 > 32767 → sat +32767
        drive(`MAMBA_PE_ADD, 0, 16'sd32000, 16'sd32000, 16'sd0, 16'sd0);
        @(posedge clk); #1;
        check(out_val, 16'sh7FFF, "T6c.ADD pos overflow → +max");

        // Report
        $display("");
        if (errors == 0)
            $display("===== TB DONE — all checks passed (errors=0) =====");
        else
            $display("===== TB DONE — %0d FAILURES =====", errors);
        $finish;
    end

    initial begin
        #10000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
