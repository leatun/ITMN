`timescale 1ns / 1ps

// ============================================================================
// BRAM_256b — true dual-port memory wrapper (READ_FIRST semantics).
//   Port A: write when we_a=1, always reads addr_a into dout_a (READ_FIRST:
//           dout_a receives OLD value when writing same address).
//   Port B: read-only, addr_b → dout_b.
//
//   Both dout_a and dout_b are registered → 1-cycle read latency.
//
//   1W + 2R same-cycle scenarios: as long as at most one port writes per cycle,
//   both dout_a and dout_b return valid reads. Enables MAC2 dual-weight fetch
//   from a single physical BRAM without a mirrored duplicate.
//
//   Consumers that don't need dout_a (single-read use case) can leave it
//   dangling — Vivado infers SDP and drops the port.
//
// Parameters:
//   ADDR_WIDTH : address bus width (controller uses 15-bit globally).
//   DATA_WIDTH : word width (256-bit for ram_main, ram_weight; same for ram_const).
//   RAM_STYLE  : "block" | "ultra" | "distributed" | "auto" — synth hint.
//   DEPTH      : actual entry count.  Use to declare a tighter array than
//                (1<<ADDR_WIDTH).  E.g. ram_main only needs ~4K entries so
//                DEPTH=4128 lets Vivado infer a tighter URAM cascade instead
//                of the full 32K the address width would imply.
//                Default keeps backward compatibility (=1<<ADDR_WIDTH).
// ============================================================================

module BRAM_256b #(
    parameter ADDR_WIDTH = 15,
    parameter DATA_WIDTH = 256,
    parameter RAM_STYLE  = "block",
    parameter DEPTH      = (1 << ADDR_WIDTH)
) (
    input clk,
    input we_a,
    input [ADDR_WIDTH-1:0] addr_a,
    input [DATA_WIDTH-1:0] din_a,
    output reg [DATA_WIDTH-1:0] dout_a,
    input [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] dout_b
);
    (* ram_style = RAM_STYLE *)
    reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];

    always @(posedge clk) begin
        if (we_a) ram[addr_a] <= din_a;
        dout_a <= ram[addr_a];   // READ_FIRST: old value when writing same addr
        dout_b <= ram[addr_b];
    end
endmodule
