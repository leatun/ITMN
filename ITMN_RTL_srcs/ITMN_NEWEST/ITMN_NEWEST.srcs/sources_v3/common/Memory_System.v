`include "_parameter.v"

// ============================================================================
// Memory_System — unified per-timestep memory.
//   ram_main   : 4128 x 256-bit URAM (scratch + INPUT/MAMBA_OUT aliased)
//   ram_weight : 4096 x 256-bit BRAM (per-block weights, block-0 max ~2528)
//   ram_weight2: mirror weight port for MAC2 dual read
//
// DMA target encoding:
//   2'd0 → ram_main
//   2'd2 → ram_weight
//   2'd3 → Const_Storage (routed at top level, not here)
// ============================================================================

module Memory_System (
    input         clk,
    input         reset,
    // Core read port
    input  [14:0] core_read_addr,
    output [255:0] core_read_data,
    // Core write port (independent from read — dual-port BRAM)
    input         core_write_en,
    input  [14:0] core_write_addr,
    input  [255:0] core_write_data,
    // Weight read ports
    input  [14:0] weight_read_addr,
    output [255:0] weight_read_data,
    input  [14:0] weight_read_addr2,
    output [255:0] weight_read_data2,
    // DMA write
    input         dma_write_en,
    input  [1:0]  dma_target,
    input  [14:0] dma_addr,
    input  [255:0] dma_wdata,
    // DMA read (target=0 main, =1 weight)
    input         dma_read_en,
    input  [1:0]  dma_rtarget,
    input  [14:0] dma_raddr,
    output [255:0] dma_rdata
);
    wire [255:0] out_ram_main, out_ram_w, out_ram_w2;

    wire we_main = (dma_write_en && dma_target == 2'd0) ||
                   (core_write_en && !dma_write_en);
    wire we_w    = (dma_write_en && dma_target == 2'd2);

    wire [14:0]  addr_main_wr = (dma_write_en && dma_target == 2'd0) ? dma_addr : core_write_addr;
    wire [255:0] din_main     = (dma_write_en && dma_target == 2'd0) ? dma_wdata : core_write_data;
    wire [14:0]  addr_main_rd = (dma_read_en && dma_rtarget == 2'd0) ? dma_raddr : core_read_addr;
    wire [14:0]  addr_w_rd    = (dma_read_en && dma_rtarget == 2'd2) ? dma_raddr : weight_read_addr;

    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(4128)) ram_main (
        .clk(clk), .we_a(we_main),
        .addr_a(addr_main_wr), .din_a(din_main),
        .addr_b(addr_main_rd), .dout_b(out_ram_main)
    );
    BRAM_256b #(.ADDR_WIDTH(14), .RAM_STYLE("block"), .DEPTH(4096)) ram_weight (
        .clk(clk), .we_a(we_w),
        .addr_a(dma_addr[13:0]), .din_a(dma_wdata),
        .addr_b(addr_w_rd[13:0]), .dout_b(out_ram_w)
    );
    BRAM_256b #(.ADDR_WIDTH(14), .RAM_STYLE("block"), .DEPTH(4096)) ram_weight2 (
        .clk(clk), .we_a(we_w),
        .addr_a(dma_addr[13:0]), .din_a(dma_wdata),
        .addr_b(weight_read_addr2[13:0]), .dout_b(out_ram_w2)
    );

    assign core_read_data    = out_ram_main;
    assign weight_read_data  = out_ram_w;
    assign weight_read_data2 = out_ram_w2;

    assign dma_rdata = (dma_rtarget == 2'd2) ? out_ram_w : out_ram_main;
endmodule
