`include "_parameter.v"

// ============================================================================
// Memory_System — unified storage block: main + weight_A + weight_B + const.
//
//   ram_main     : URAM, 4128 × 256b, 1R + 1W (dual-port URAM).
//                  Scratch + INPUT/MAMBA_OUT aliased.
//   ram_weight_A : BRAM, W_MEM_DEPTH × 256b, TDP mode — 1W + 2R same-cycle.
//                  Holds even-group PP weights + SMALLS single-copy
//                  (W_DW, W_A, evens of W_INPROJ_X/Z, W_XPROJ, W_OUTPROJ).
//                  Served to Cluster1.
//   ram_weight_B : BRAM, W_MEM_DEPTH × 256b, TDP mode — 1W + 2R same-cycle.
//                  Holds ONLY odd-group PP weights (W_INPROJ_X/Z, W_XPROJ,
//                  W_OUTPROJ). Content = complement of bank_A's PP section;
//                  NOT a mirror of bank_A. Served to Cluster2.
//                  DMA loader must route weight rows by group parity.
//   ram_const    : BRAM, 128 × 256b, 1R + 1W (small, holds per-block biases
//                  / RMSNorm gamma / D_param). Loaded via DMA target=3'd3.
//
// DMA target encoding (extended 2→3-bit to accommodate bank_B):
//   3'd0 → ram_main
//   3'd2 → ram_weight_A   (even-group PP weights + SMALLS)
//   3'd3 → ram_const
//   3'd4 → ram_weight_B   (odd-group PP weights)
//
// DMA read target (`dma_rtarget`, same 3-bit encoding).
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
    // Weight bank A read ports (Cluster1) — 2R same-cycle for MAC2
    input  [14:0] weight_read_addr,
    output [255:0] weight_read_data,
    input  [14:0] weight_read_addr2,
    output [255:0] weight_read_data2,
    // Weight bank B read ports (Cluster2) — 2R same-cycle for MAC2
    input  [14:0] weight_read_addr3,
    output [255:0] weight_read_data3,
    input  [14:0] weight_read_addr4,
    output [255:0] weight_read_data4,
    // Const read port (ram_const)
    input  [14:0] const_read_addr,
    output [255:0] const_read_data,
    // DMA write (target selects RAM) — 3-bit
    input         dma_write_en,
    input  [2:0]  dma_target,
    input  [14:0] dma_addr,
    input  [255:0] dma_wdata,
    // DMA read
    input         dma_read_en,
    input  [2:0]  dma_rtarget,
    input  [14:0] dma_raddr,
    output [255:0] dma_rdata
);
    wire [255:0] out_ram_main;
    wire [255:0] out_ram_wA_a, out_ram_wA_b;
    wire [255:0] out_ram_wB_a, out_ram_wB_b;
    wire [255:0] out_ram_c;

    wire we_main   = (dma_write_en && dma_target == 3'd0) ||
                     (core_write_en && !dma_write_en);
    wire we_wA     = (dma_write_en && dma_target == 3'd2);
    wire we_wB     = (dma_write_en && dma_target == 3'd4);
    wire we_const  = (dma_write_en && dma_target == 3'd3);

    wire [14:0]  addr_main_wr = (dma_write_en && dma_target == 3'd0) ? dma_addr : core_write_addr;
    wire [255:0] din_main     = (dma_write_en && dma_target == 3'd0) ? dma_wdata : core_write_data;
    wire [14:0]  addr_main_rd = (dma_read_en && dma_rtarget == 3'd0) ? dma_raddr : core_read_addr;

    // Bank A port B: primary W1 read (Cluster1), also carries DMA read-back for target=2
    wire [14:0]  addr_wA_b_rd = (dma_read_en && dma_rtarget == 3'd2) ? dma_raddr : weight_read_addr;
    // Bank A port A: DMA write OR secondary W2 read (Cluster1)
    wire [12:0]  addr_wA_a    = we_wA ? dma_addr[12:0] : weight_read_addr2[12:0];

    // Bank B port B: primary W1 read (Cluster2), also carries DMA read-back for target=4
    wire [14:0]  addr_wB_b_rd = (dma_read_en && dma_rtarget == 3'd4) ? dma_raddr : weight_read_addr3;
    // Bank B port A: DMA write OR secondary W2 read (Cluster2)
    wire [12:0]  addr_wB_a    = we_wB ? dma_addr[12:0] : weight_read_addr4[12:0];

    // Const read: DMA readback OR controller read
    wire [14:0]  addr_c_rd    = (dma_read_en && dma_rtarget == 3'd3) ? dma_raddr : const_read_addr;

    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(4128)) ram_main (
        .clk    (clk),
        .we_a   (we_main),
        .addr_a (addr_main_wr),
        .din_a  (din_main),
        .addr_b (addr_main_rd),
        .dout_b (out_ram_main)
    );

    BRAM_256b #(.ADDR_WIDTH(13), .RAM_STYLE("block"), .DEPTH(`W_MEM_DEPTH)) ram_weight_A (
        .clk    (clk),
        .we_a   (we_wA),
        .addr_a (addr_wA_a),
        .din_a  (dma_wdata),
        .dout_a (out_ram_wA_a),
        .addr_b (addr_wA_b_rd[12:0]),
        .dout_b (out_ram_wA_b)
    );

    BRAM_256b #(.ADDR_WIDTH(13), .RAM_STYLE("block"), .DEPTH(`W_MEM_DEPTH_B)) ram_weight_B (
        .clk    (clk),
        .we_a   (we_wB),
        .addr_a (addr_wB_a),
        .din_a  (dma_wdata),
        .dout_a (out_ram_wB_a),
        .addr_b (addr_wB_b_rd[12:0]),
        .dout_b (out_ram_wB_b)
    );

    BRAM_256b #(.ADDR_WIDTH(7), .RAM_STYLE("block"), .DEPTH(128)) ram_const (
        .clk    (clk),
        .we_a   (we_const),
        .addr_a (dma_addr[6:0]),
        .din_a  (dma_wdata),
        .addr_b (addr_c_rd[6:0]),
        .dout_b (out_ram_c)
    );

    assign core_read_data     = out_ram_main;
    assign weight_read_data   = out_ram_wA_b;   // Cluster1 W1
    assign weight_read_data2  = out_ram_wA_a;   // Cluster1 W2
    assign weight_read_data3  = out_ram_wB_b;   // Cluster2 W1
    assign weight_read_data4  = out_ram_wB_a;   // Cluster2 W2
    assign const_read_data    = out_ram_c;

    assign dma_rdata = (dma_rtarget == 3'd2) ? out_ram_wA_b :
                       (dma_rtarget == 3'd4) ? out_ram_wB_b :
                       (dma_rtarget == 3'd3) ? out_ram_c    :
                                                out_ram_main;
endmodule
