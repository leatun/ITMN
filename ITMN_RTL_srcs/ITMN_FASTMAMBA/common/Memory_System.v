`include "_parameter.v"

// ============================================================================
// Memory_System — unified storage block: main + weight_A + weight_B + const.
//
//   ram_main     : URAM, 4128 × 256b, TDP — Port A (c2 read OR c1 write),
//                  Port B (c1 read). Phase 5 refactor: exposed port A as
//                  dedicated c2 read (no more mutex). c1 has write priority;
//                  c2 read preempted for 1 cyc when c1 writes.
//   ram_weight_A : BRAM, W_MEM_DEPTH × 256b, TDP mode — 1W + 2R same-cycle.
//                  Cluster1 weights.
//   ram_weight_B : BRAM, W_MEM_DEPTH × 256b, TDP mode — 1W + 2R same-cycle.
//                  Cluster2 weights.
//   ram_const    : BRAM, 128 × 256b, TDP — Port A (c2 read OR DMA write),
//                  Port B (c1 read).
//
// DMA target encoding (3-bit):
//   3'd0 → ram_main
//   3'd2 → ram_weight_A   (even-group PP weights + SMALLS)
//   3'd3 → ram_const
//   3'd4 → ram_weight_B   (odd-group PP weights)
// ============================================================================

module Memory_System (
    input         clk,
    input         reset,
    // Cluster1 core read port (ram_main port B)
    input  [14:0] core_read_addr,
    output [255:0] core_read_data,
    // Cluster2 core read port (ram_main port A when not writing)
    input  [14:0] core_read_addr2,
    output [255:0] core_read_data2,
    // Unified write port (ram_main port A — c1/c2 muxed at controller level)
    input         core_write_en,
    input  [14:0] core_write_addr,
    input  [255:0] core_write_data,
    // Weight bank A read ports (Cluster1)
    input  [14:0] weight_read_addr,
    output [255:0] weight_read_data,
    input  [14:0] weight_read_addr2,
    output [255:0] weight_read_data2,
    // Weight bank B read ports (Cluster2)
    input  [14:0] weight_read_addr3,
    output [255:0] weight_read_data3,
    input  [14:0] weight_read_addr4,
    output [255:0] weight_read_data4,
    // Const read ports — c1 (port B), c2 (port A when not DMA writing)
    input  [14:0] const_read_addr,
    output [255:0] const_read_data,
    input  [14:0] const_read_addr2,
    output [255:0] const_read_data2,
    // DMA write
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
    wire [255:0] out_ram_main;      // port B → c1
    wire [255:0] out_ram_main2;     // port A → c2
    wire [255:0] out_ram_wA_a, out_ram_wA_b;
    wire [255:0] out_ram_wB_a, out_ram_wB_b;
    wire [255:0] out_ram_c;         // port B → c1
    wire [255:0] out_ram_c2;        // port A → c2

    wire we_main   = (dma_write_en && dma_target == 3'd0) ||
                     (core_write_en && !dma_write_en);
    wire we_wA     = (dma_write_en && dma_target == 3'd2);
    wire we_wB     = (dma_write_en && dma_target == 3'd4);
    wire we_const  = (dma_write_en && dma_target == 3'd3);

    // ram_main writes (both mirror copies see same write).
    wire [14:0]  addr_main_wr = (dma_write_en && dma_target == 3'd0) ? dma_addr : core_write_addr;
    wire [255:0] din_main     = (dma_write_en && dma_target == 3'd0) ? dma_wdata : core_write_data;

    // ram_main port B: c1 read (or DMA read-back for target=0)
    wire [14:0]  addr_main_b  = (dma_read_en && dma_rtarget == 3'd0) ? dma_raddr : core_read_addr;

    // ram_const port A: write when we_const=1 (DMA), else c2 read
    wire [6:0]   addr_const_a = we_const ? dma_addr[6:0] : const_read_addr2[6:0];
    // ram_const port B: c1 read (or DMA read-back for target=3)
    wire [14:0]  addr_c_rd_b  = (dma_read_en && dma_rtarget == 3'd3) ? dma_raddr : const_read_addr;

    // Weight banks (unchanged from Phase 5)
    wire [14:0]  addr_wA_b_rd = (dma_read_en && dma_rtarget == 3'd2) ? dma_raddr : weight_read_addr;
    wire [12:0]  addr_wA_a    = we_wA ? dma_addr[12:0] : weight_read_addr2[12:0];
    wire [14:0]  addr_wB_b_rd = (dma_read_en && dma_rtarget == 3'd4) ? dma_raddr : weight_read_addr3;
    wire [12:0]  addr_wB_a    = we_wB ? dma_addr[12:0] : weight_read_addr4[12:0];

    // Phase 6 fix: MIRROR ram_main into two URAM-friendly SDP copies.
    // Both copies get identical writes (write bus fanout to both). c1/c2 each
    // read from own copy via port B. No TDP RW conflict -> URAM inferred cleanly.
    // Cost: +8 URAM (16 total for ram_main). Saves 36 BRAM (Phase 5 fell back to BRAM).
    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(4128)) ram_main_c1 (
        .clk    (clk),
        .we_a   (we_main),
        .addr_a (addr_main_wr),
        .din_a  (din_main),
        .addr_b (addr_main_b),
        .dout_b (out_ram_main)
    );

    BRAM_256b #(.ADDR_WIDTH(15), .RAM_STYLE("ultra"), .DEPTH(4128)) ram_main_c2 (
        .clk    (clk),
        .we_a   (we_main),                    // same write bus (mirror)
        .addr_a (addr_main_wr),
        .din_a  (din_main),
        .addr_b (core_read_addr2),
        .dout_b (out_ram_main2)
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
        .addr_a (addr_const_a),
        .din_a  (dma_wdata),
        .dout_a (out_ram_c2),       // c2 const read (Phase 5)
        .addr_b (addr_c_rd_b[6:0]),
        .dout_b (out_ram_c)         // c1 const read
    );

    assign core_read_data     = out_ram_main;
    assign core_read_data2    = out_ram_main2;
    assign weight_read_data   = out_ram_wA_b;
    assign weight_read_data2  = out_ram_wA_a;
    assign weight_read_data3  = out_ram_wB_b;
    assign weight_read_data4  = out_ram_wB_a;
    assign const_read_data    = out_ram_c;
    assign const_read_data2   = out_ram_c2;

    assign dma_rdata = (dma_rtarget == 3'd2) ? out_ram_wA_b :
                       (dma_rtarget == 3'd4) ? out_ram_wB_b :
                       (dma_rtarget == 3'd3) ? out_ram_c    :
                                                out_ram_main;
endmodule
