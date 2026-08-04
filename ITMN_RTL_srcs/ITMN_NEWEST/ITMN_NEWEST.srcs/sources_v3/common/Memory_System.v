`include "_parameter.v"

// ============================================================================
// Memory_System — unified storage block: main + weight_A + weight_B + const.
//
// Phase 6 backport (from ITMN_FASTMAMBA):
//   * ram_main mirrored into 2 SDP URAM copies (ram_main_c1, ram_main_c2).
//     Fanout write, dedicated port_b reads per cluster. Avoids TDP RW pattern
//     which would fall back to BRAM36 (~36 tiles); cost: +8 URAM but stays URAM.
//   * ram_weight split by group parity into 2 physical BRAMs:
//       ram_weight_A: SMALLS (W_DW, W_XPROJ, W_DTPROJ, W_A) + EVEN-group PP weights.
//       ram_weight_B: ODD-group PP weights only (no SMALLS).
//     Cluster1 reads bank_A (W1+W2 via TDP 2R); cluster2 reads bank_B (2R).
//     In Phase 1 (c2 idle) bank_B stays empty — TB loads all weights via target=2.
//   * ram_const now dual-port: port_b = c1 read, port_a = c2 read OR DMA write.
//
//   ram_main_c1 : URAM, 4128 × 256b, SDP — port_a = c1/DMA write; port_b = c1 read.
//   ram_main_c2 : URAM, 4128 × 256b, SDP — port_a = mirror write (same as c1);
//                                            port_b = c2 read.
//   ram_weight_A: BRAM, W_MEM_DEPTH   × 256b, TDP — port_a (2nd read W2 or DMA write);
//                                                   port_b (primary read W1).
//   ram_weight_B: BRAM, W_MEM_DEPTH_B × 256b, TDP — port_a (2nd read W2 or DMA write);
//                                                   port_b (primary read W1).
//   ram_const   : BRAM, 128 × 256b, TDP — port_a (c2 read OR DMA write);
//                                          port_b (c1 read).
//
// Weight layout bank_A (unchanged from baseline, see _parameter.v):
//   [0..1216)      SMALLS resident (W_DW, W_XPROJ, W_DTPROJ, W_A)
//   [1216..3264)   Slot X   — W_INPROJ_X permanent (even-group half when Phase 2 splits)
//   [3264..5312)   Slot Z   — W_INPROJ_Z permanent
//   [5312..7360)   Slot OUT — W_OUTPROJ permanent
//   [7360..8192)   spare
//
// Weight layout bank_B (Phase 2+ only — mirrors odd-group PP weight slots at
// SAME raw g addresses as bank_A). Bank_A stays authoritative for all-group
// weights + SMALLS + M4 W_XPROJ. Bank_B receives DMA writes ONLY for odd
// groups of W_INPROJ_X/W_INPROJ_Z/W_OUTPROJ. Same DEPTH (8192) as bank_A so
// c2 uses identical address formula as c1 (raw g offset, no dense remap).
//
// DMA target encoding (3-bit):
//   3'd0 → ram_main
//   3'd2 → ram_weight_A   (SMALLS + even-group PP weights)
//   3'd3 → ram_const
//   3'd4 → ram_weight_B   (odd-group PP weights)
// ============================================================================

module Memory_System (
    input         clk,
    input         reset,
    // Cluster1 core read port (ram_main_c1 port_b)
    input  [14:0] core_read_addr,
    output [255:0] core_read_data,
    // Cluster2 core read port (ram_main_c2 port_b)
    input  [14:0] core_read_addr2,
    output [255:0] core_read_data2,
    // Unified core write port (fanout to both mirrors — c1/c2 muxed at controller level)
    input         core_write_en,
    input  [14:0] core_write_addr,
    input  [255:0] core_write_data,
    // Weight bank A read ports (Cluster1) — TDP 2R
    input  [14:0] weight_read_addr,
    output [255:0] weight_read_data,
    input  [14:0] weight_read_addr2,
    output [255:0] weight_read_data2,
    // Weight bank B read ports (Cluster2) — TDP 2R
    input  [14:0] weight_read_addr3,
    output [255:0] weight_read_data3,
    input  [14:0] weight_read_addr4,
    output [255:0] weight_read_data4,
    // Const read ports — c1 (port_b), c2 (port_a when not DMA writing)
    input  [14:0] const_read_addr,
    output [255:0] const_read_data,
    input  [14:0] const_read_addr2,
    output [255:0] const_read_data2,
    // DMA write (3-bit target)
    input         dma_write_en,
    input  [2:0]  dma_target,
    input  [14:0] dma_addr,
    input  [255:0] dma_wdata,
    // DMA read (rtarget: 0=main, 2=wA, 3=const, 4=wB)
    input         dma_read_en,
    input  [2:0]  dma_rtarget,
    input  [14:0] dma_raddr,
    output [255:0] dma_rdata
);
    wire [255:0] out_ram_main;      // port_b → c1
    wire [255:0] out_ram_main2;     // port_b → c2
    wire [255:0] out_ram_wA_a, out_ram_wA_b;
    wire [255:0] out_ram_wB_a, out_ram_wB_b;
    wire [255:0] out_ram_c;         // port_b → c1
    wire [255:0] out_ram_c2;        // port_a → c2

    wire we_main   = (dma_write_en && dma_target == 3'd0) ||
                     (core_write_en && !dma_write_en);
    wire we_wA     = (dma_write_en && dma_target == 3'd2);
    wire we_wB     = (dma_write_en && dma_target == 3'd4);
    wire we_const  = (dma_write_en && dma_target == 3'd3);

    // ram_main writes (both mirror copies see identical write bus).
    wire [14:0]  addr_main_wr = (dma_write_en && dma_target == 3'd0) ? dma_addr : core_write_addr;
    wire [255:0] din_main     = (dma_write_en && dma_target == 3'd0) ? dma_wdata : core_write_data;

    // ram_main port_b: c1 read (or DMA read-back for target=0)
    wire [14:0]  addr_main_b  = (dma_read_en && dma_rtarget == 3'd0) ? dma_raddr : core_read_addr;

    // ram_const port_a: DMA write when we_const, else c2 read
    wire [6:0]   addr_const_a = we_const ? dma_addr[6:0] : const_read_addr2[6:0];
    // ram_const port_b: c1 read (or DMA read-back for target=3)
    wire [14:0]  addr_c_rd_b  = (dma_read_en && dma_rtarget == 3'd3) ? dma_raddr : const_read_addr;

    // Weight bank A: port_b = W1 read; port_a = W2 read OR DMA write.
    wire [14:0]  addr_wA_b_rd = (dma_read_en && dma_rtarget == 3'd2) ? dma_raddr : weight_read_addr;
    wire [12:0]  addr_wA_a    = we_wA ? dma_addr[12:0] : weight_read_addr2[12:0];
    // Weight bank B: DENSE odd-only (Phase 6 opt) — DEPTH=4096, ADDR_W=12.
    wire [14:0]  addr_wB_b_rd = (dma_read_en && dma_rtarget == 3'd4) ? dma_raddr : weight_read_addr3;
    wire [11:0]  addr_wB_a    = we_wB ? dma_addr[11:0] : weight_read_addr4[11:0];

    // Mirror ram_main: 2 SDP URAM copies with fanout write.
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

    BRAM_256b #(.ADDR_WIDTH(12), .RAM_STYLE("block"), .DEPTH(`W_MEM_DEPTH_B)) ram_weight_B (
        .clk    (clk),
        .we_a   (we_wB),
        .addr_a (addr_wB_a),
        .din_a  (dma_wdata),
        .dout_a (out_ram_wB_a),
        .addr_b (addr_wB_b_rd[11:0]),
        .dout_b (out_ram_wB_b)
    );

    BRAM_256b #(.ADDR_WIDTH(7), .RAM_STYLE("distributed"), .DEPTH(128)) ram_const (
        .clk    (clk),
        .we_a   (we_const),
        .addr_a (addr_const_a),
        .din_a  (dma_wdata),
        .dout_a (out_ram_c2),       // c2 const read
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
