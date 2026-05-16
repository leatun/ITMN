`timescale 1ns / 1ps

module BRAM_256b #(parameter ADDR_WIDTH = 15, parameter DATA_WIDTH = 256) (
    input clk,
    input we_a,
    input [ADDR_WIDTH-1:0] addr_a,
    input [DATA_WIDTH-1:0] din_a,
    input [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] dout_b
);
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] ram [0:(1<<ADDR_WIDTH)-1];

    always @(posedge clk) if (we_a) ram[addr_a] <= din_a;
    always @(posedge clk) dout_b <= ram[addr_b];
endmodule