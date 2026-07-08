`include "_parameter.v"

// ============================================================================
// Const_Storage — single hierarchy block holding every read-only / config
// storage in the design.  Three categories:
//
//   1. Activation LUTs (16 parallel lanes, combinational):
//        - Silu_LUT     : x_in (Q4.11) → silu(x)         (S_M3, S_M7)
//        - Softplus_LUT : x_in (Q4.11) → softplus(x)     (S_M5)
//        - Exp_LUT      : x_in (Q4.11) → exp(x)          (S_M6A_DA)
//      Each lane is its own instance because distributed LUTRAM has a single
//      read port — 16 parallel reads require 16 physical copies.
//
//   2. RSqrt ROM (8K × 16-bit, BRAM-inferred):
//        Indexed by `rsqrt_idx` (= norm_rom_idx) once per timestep in the
//        RMSNorm mean state to obtain S = rsqrt(mean(x²)).
//
//   3. ram_const (64 × 256-bit, distributed-LUT-inferred):
//        Per-block bias / BN scale / BN shift / depthwise bias / dt bias /
//        RMSNorm gamma.  Loaded by host DMA before each block run
//        (dma_target == 2'd3), then read combinationally-then-registered by
//        the FSM via `const_read_addr` / `const_read_data` (1-cycle latency).
//        Also reachable via DMA read (dma_rtarget == 2'd3) for host verify.
//
// Tables 1 & 2 are init'd from golden_all/{silu,softplus,exp}_lut.txt and
// golden_all/rsqrt_q97.txt at synth/sim time.  ram_const is empty at reset.
// ============================================================================

module Const_Storage (
    input  wire        clk,

    // ---- Activation LUTs (combinational, 16-lane flat) ----
    input  wire [255:0] silu_in_flat,
    input  wire [255:0] sp_in_flat,
    input  wire [255:0] exp_in_flat,
    output wire [255:0] silu_out_flat,
    output wire [255:0] sp_out_flat,
    output wire [255:0] exp_out_flat,

    // ---- RSqrt ROM ----
    input  wire [12:0] rsqrt_idx,
    output wire [15:0] rsqrt_data,

    // ---- ram_const: DMA write port (target == 2'd3) ----
    input  wire        dma_write_en,
    input  wire [1:0]  dma_target,
    input  wire [14:0] dma_addr,
    input  wire [255:0] dma_wdata,
    // ---- ram_const: core read port ----
    input  wire [14:0] const_read_addr,
    output wire [255:0] const_read_data,
    // ---- ram_const: DMA read port (rtarget == 2'd3) ----
    input  wire        dma_read_en,
    input  wire [1:0]  dma_rtarget,
    input  wire [14:0] dma_raddr,
    output wire [255:0] dma_rdata_const
);

    // ---- 16-lane activation LUT generate ----
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : ACT_LANES
            Silu_LUT     u_silu (
                .x_in     (silu_in_flat [gi*16 +: 16]),
                .silu_out (silu_out_flat[gi*16 +: 16])
            );
            Softplus_LUT u_softplus (
                .x_in         (sp_in_flat  [gi*16 +: 16]),
                .softplus_out (sp_out_flat [gi*16 +: 16])
            );
            Exp_LUT      u_exp (
                .x_in    (exp_in_flat [gi*16 +: 16]),
                .exp_out (exp_out_flat[gi*16 +: 16])
            );
        end
    endgenerate

    // ---- RSqrt ROM ----
    RSqrt_ROM u_rsqrt_rom (
        .idx  (rsqrt_idx),
        .data (rsqrt_data)
    );

    // ---- ram_const (128 × 256, distributed-LUT-inferred) ----
    // Expanded from 64 → 128 to host D2 encoder + FC bias regions
    // (see ITM_CONTROLLER_v3.v: C_ENC_BIAS=64, C_FC_BIAS=68).  D1 builds use
    // only entries [0..63] so the upper half stays at 0 — backward compatible.
    //
    // Write: DMA when target==3, addr_a = dma_addr[6:0]
    // Read : either DMA (rtarget==3 → dma_raddr) or controller (const_read_addr),
    //        registered output via BRAM_256b (1-cycle latency).
    wire        we_const = dma_write_en && (dma_target == 2'd3);
    wire [14:0] addr_b_const = (dma_read_en && dma_rtarget == 2'd3) ? dma_raddr
                                                                    : const_read_addr;
    wire [255:0] out_ram_const;
    BRAM_256b #(.ADDR_WIDTH(7), .RAM_STYLE("block")) ram_const (
        .clk    (clk),
        .we_a   (we_const),
        .addr_a (dma_addr[6:0]),
        .din_a  (dma_wdata),
        .addr_b (addr_b_const[6:0]),
        .dout_b (out_ram_const)
    );
    assign const_read_data = out_ram_const;
    assign dma_rdata_const = out_ram_const;

endmodule
