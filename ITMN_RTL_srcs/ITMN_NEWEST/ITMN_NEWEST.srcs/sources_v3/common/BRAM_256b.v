`timescale 1ns / 1ps

// ============================================================================
// BRAM_256b — simple dual-port memory wrapper.
//   addr_a/we_a/din_a  : write port  (registered)
//   addr_b/dout_b      : read  port  (registered output, 1-cycle latency)
//
// Parameters:
//   ADDR_WIDTH : address bus width (controller uses 15-bit globally).
//   DATA_WIDTH : word width (256-bit for ram_a/b, ram_w; same for ram_const).
//   RAM_STYLE  : "block" | "ultra" | "distributed" | "auto" — synth hint.
//   DEPTH      : actual entry count.  Use to declare a tighter array than
//                (1<<ADDR_WIDTH).  E.g. ram_a/b only need 20K entries
//                (compact map peak 17256/19000), so DEPTH=20480 lets Vivado
//                infer 5-deep URAM cascade × 4-wide = 20 URAM/bank instead
//                of the 8-deep × 4-wide = 32 URAM the full 32K would imply.
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
    input [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] dout_b
);
    (* ram_style = RAM_STYLE *)
    reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];

    always @(posedge clk) if (we_a) ram[addr_a] <= din_a;
    always @(posedge clk) dout_b <= ram[addr_b];
endmodule
