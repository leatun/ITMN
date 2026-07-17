`include "_parameter.v"

// ============================================================================
// Memory_System — unified storage block: main + weight + const.
//
//   ram_main   : URAM, 4128 × 256b, 1R + 1W (dual-port URAM).
//                Scratch + INPUT/MAMBA_OUT aliased.
//   ram_weight : BRAM, 8192 × 256b, TDP mode — 1W + 2R same-cycle.
//                Port A dual-role: DMA write when we_w=1, else 2nd read (W2).
//                Port B: always the primary read (W1).
//                No mirror — MAC2 dual weight fetch served from a single BRAM.
//                Runtime writes forbidden (all preload happens before start).
//   ram_const  : BRAM, 128 × 256b, 1R + 1W (small, holds per-block biases
//                / RMSNorm gamma / D_param). Loaded via DMA target=3.
//
// Weight layout (see _parameter.v, W_MEM_DEPTH=8192):
//   [0..1216)      SMALLS resident (W_DW, W_XPROJ, W_DTPROJ, W_A)
//   [1216..3264)   Slot X   — W_INPROJ_X permanent
//   [3264..5312)   Slot Z   — W_INPROJ_Z permanent
//   [5312..7360)   Slot OUT — W_OUTPROJ permanent (sized for B4 d_inner=256)
//   [7360..8192)   spare
//
// Const layout (see _parameter.v):
//   [0..8)         C_W_NORM_BASE   (RMSNorm gamma)
//   [8..24)        C_B_DW_BASE     (depthwise conv bias)
//   [24..40)       C_B_DT_BASE     (dt bias)
//   [40..56)       C_D_PARAM_BASE  (D param)
//
// DMA target encoding:
//   2'd0 → ram_main
//   2'd2 → ram_weight
//   2'd3 → ram_const
// ============================================================================

module Memory_System (
    input         clk,
    input         reset,
    // Core read port (ram_main)
    input  [14:0] core_read_addr,
    output [255:0] core_read_data,
    // Core write port (ram_main)
    input         core_write_en,
    input  [14:0] core_write_addr,
    input  [255:0] core_write_data,
    // Weight read ports (both served from single ram_weight via TDP)
    input  [14:0] weight_read_addr,
    output [255:0] weight_read_data,
    input  [14:0] weight_read_addr2,
    output [255:0] weight_read_data2,
    // Const read port (ram_const)
    input  [14:0] const_read_addr,
    output [255:0] const_read_data,
    // DMA write (target selects RAM)
    input         dma_write_en,
    input  [1:0]  dma_target,
    input  [14:0] dma_addr,
    input  [255:0] dma_wdata,
    // DMA read (rtarget=0 main, =2 weight, =3 const)
    input         dma_read_en,
    input  [1:0]  dma_rtarget,
    input  [14:0] dma_raddr,
    output [255:0] dma_rdata
);
    wire [255:0] out_ram_main, out_ram_w_a, out_ram_w_b, out_ram_c;

    wire we_main  = (dma_write_en && dma_target == 2'd0) ||
                    (core_write_en && !dma_write_en);
    wire we_w     = (dma_write_en && dma_target == 2'd2);
    wire we_const = (dma_write_en && dma_target == 2'd3);

    wire [14:0]  addr_main_wr = (dma_write_en && dma_target == 2'd0) ? dma_addr : core_write_addr;
    wire [255:0] din_main     = (dma_write_en && dma_target == 2'd0) ? dma_wdata : core_write_data;
    wire [14:0]  addr_main_rd = (dma_read_en && dma_rtarget == 2'd0) ? dma_raddr : core_read_addr;

    // Weight port B: primary read (W1), also carries DMA read-back
    wire [14:0]  addr_w_b_rd  = (dma_read_en && dma_rtarget == 2'd2) ? dma_raddr : weight_read_addr;
    // Weight port A: DMA write OR secondary weight read (W2)
    wire [12:0]  addr_w_a     = we_w ? dma_addr[12:0] : weight_read_addr2[12:0];

    // Const read: DMA readback OR controller read
    wire [14:0]  addr_c_rd    = (dma_read_en && dma_rtarget == 2'd3) ? dma_raddr : const_read_addr;

    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(4128)) ram_main (
        .clk    (clk),
        .we_a   (we_main),
        .addr_a (addr_main_wr),
        .din_a  (din_main),
        .addr_b (addr_main_rd),
        .dout_b (out_ram_main)
    );

    BRAM_256b #(.ADDR_WIDTH(13), .RAM_STYLE("block"), .DEPTH(`W_MEM_DEPTH)) ram_weight (
        .clk    (clk),
        .we_a   (we_w),
        .addr_a (addr_w_a),
        .din_a  (dma_wdata),
        .dout_a (out_ram_w_a),
        .addr_b (addr_w_b_rd[12:0]),
        .dout_b (out_ram_w_b)
    );

    BRAM_256b #(.ADDR_WIDTH(7), .RAM_STYLE("block"), .DEPTH(128)) ram_const (
        .clk    (clk),
        .we_a   (we_const),
        .addr_a (dma_addr[6:0]),
        .din_a  (dma_wdata),
        .addr_b (addr_c_rd[6:0]),
        .dout_b (out_ram_c)
    );

    assign core_read_data    = out_ram_main;
    assign weight_read_data  = out_ram_w_b;
    assign weight_read_data2 = out_ram_w_a;
    assign const_read_data   = out_ram_c;

    assign dma_rdata = (dma_rtarget == 2'd2) ? out_ram_w_b :
                       (dma_rtarget == 2'd3) ? out_ram_c   :
                                                out_ram_main;
endmodule
