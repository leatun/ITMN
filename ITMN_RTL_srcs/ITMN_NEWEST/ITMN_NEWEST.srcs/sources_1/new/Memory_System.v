`include "_parameter.v"

// ============================================================================
// Memory_System — bulk working memory (read/write storage).
//   ram_a   : 32K x 256-bit, URAM (Mamba working storage; ~30K word used)
//   ram_b   : 32K x 256-bit, URAM (intermediate buffers)
//   ram_w   : 16K x 256-bit, BRAM (weights for one block; <14K word used)
//
// Controller drives 15-bit addresses (m_rd_addr, m_wr_addr, w_rd_addr).
// The narrower BRAM (ram_w) ignores upper bits via explicit slicing.
// Caller MUST guarantee addresses fit the BRAM range, otherwise wrap-around
// will silently corrupt data.
//
// Read-only constant data (per-block bias/scale/shift, activation LUTs,
// rsqrt ROM) lives in Const_Storage, not here.  DMA target encoding:
//   2'd0 → ram_a    (here)
//   2'd1 → ram_b    (here)
//   2'd2 → ram_w    (here)
//   2'd3 → ram_const (Const_Storage)
// Targets 0-2 are handled below; target 3 is ignored here and routed to
// Const_Storage at the top level.
// ============================================================================

module Memory_System (
    input clk, input reset,
    input bank_sel,
    input  [14:0] core_read_addr,  output [255:0] core_read_data,
    input         core_write_en,   input [14:0] core_write_addr, input [255:0] core_write_data,
    input  [14:0] weight_read_addr, output [255:0] weight_read_data,
    input         dma_write_en,
    input  [1:0]  dma_target,
    input  [14:0] dma_addr,
    input  [255:0] dma_wdata,
    // DMA READ interface (host reads BRAM back).  rtarget==3 (ram_const) is
    // not present here; that path is owned by Const_Storage.
    input         dma_read_en,
    input  [1:0]  dma_rtarget,
    input  [14:0] dma_raddr,
    output [255:0] dma_rdata
);
    wire [255:0] out_ram_a, out_ram_b, out_ram_w;

    // ---- Write enables ----
    wire we_a = (dma_write_en && dma_target==2'd0) || (core_write_en && bank_sel==1 && !dma_write_en);
    wire we_b = (dma_write_en && dma_target==2'd1) || (core_write_en && bank_sel==0 && !dma_write_en);
    wire we_w = (dma_write_en && dma_target==2'd2);

    // ---- Write address/data muxes ----
    wire [14:0]  addr_a_wr = (dma_write_en && dma_target==0) ? dma_addr : core_write_addr;
    wire [255:0] din_a     = (dma_write_en && dma_target==0) ? dma_wdata : core_write_data;
    wire [14:0]  addr_b_wr = (dma_write_en && dma_target==1) ? dma_addr : core_write_addr;
    wire [255:0] din_b     = (dma_write_en && dma_target==1) ? dma_wdata : core_write_data;

    // ---- Read address muxes (DMA read takes over when active) ----
    wire [14:0] addr_b_ram_a = (dma_read_en && dma_rtarget==2'd0) ? dma_raddr :
                               (bank_sel==0 ? core_read_addr : 15'd0);
    wire [14:0] addr_b_ram_b = (dma_read_en && dma_rtarget==2'd1) ? dma_raddr :
                               (bank_sel==1 ? core_read_addr : 15'd0);
    wire [14:0] addr_b_ram_w = (dma_read_en && dma_rtarget==2'd2) ? dma_raddr : weight_read_addr;

    // ============================================================
    //  ram_a, ram_b: 20K x 256 → URAM (5-deep × 4-wide = 20 URAM/bank)
    //  Address bus stays 15-bit (controller still drives full range) but
    //  array is declared at exact depth so Vivado infers fewer URAM tiles.
    //  Compact map peaks: bank A 17256, bank B 19000 → 20K covers both
    //  with the 4K URAM granularity headroom.
    // ============================================================
    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(20480)) ram_a (
        .clk(clk), .we_a(we_a),
        .addr_a(addr_a_wr),       .din_a(din_a),
        .addr_b(addr_b_ram_a),    .dout_b(out_ram_a)
    );
    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(20480)) ram_b (
        .clk(clk), .we_a(we_b),
        .addr_a(addr_b_wr),       .din_a(din_b),
        .addr_b(addr_b_ram_b),    .dout_b(out_ram_b)
    );

    // ============================================================
    //  ram_weight: 16K x 256 → BRAM. Drop upper bit of 15-bit address.
    // ============================================================
    BRAM_256b #(.ADDR_WIDTH(14), .RAM_STYLE("block")) ram_weight (
        .clk(clk), .we_a(we_w),
        .addr_a(dma_addr[13:0]),       .din_a(dma_wdata),
        .addr_b(addr_b_ram_w[13:0]),   .dout_b(out_ram_w)
    );

    // ---- Outputs ----
    assign weight_read_data = out_ram_w;
    assign core_read_data   = (bank_sel == 0) ? out_ram_a : out_ram_b;

    // dma_rdata: 3-way mux for targets 0-2.  rtarget==3 falls back to ram_a
    // (don't-care — host should select between this and Const_Storage's
    // dma_rdata_const at the top level via dma_rtarget).
    assign dma_rdata = (dma_rtarget == 2'd0) ? out_ram_a :
                       (dma_rtarget == 2'd1) ? out_ram_b :
                       (dma_rtarget == 2'd2) ? out_ram_w :
                                               out_ram_a;
endmodule
