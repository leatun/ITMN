`timescale 1ns / 1ps

module BRAM_256b
#(
    parameter ADDR_WIDTH = 15,
    parameter DATA_WIDTH = 256 
)
(
    input clk,
    
    // Port A: (Write)
    input we_a,                  // Write Enable
    input [ADDR_WIDTH-1:0] addr_a,
    input [DATA_WIDTH-1:0] din_a,
    
    // Port B: (Read)
    input [ADDR_WIDTH-1:0] addr_b,
    output reg [DATA_WIDTH-1:0] dout_b
);

    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] ram [0 : (1<<ADDR_WIDTH)-1];

    // Logic Write (Port A)
    always @(posedge clk) begin
        if (we_a) begin
            ram[addr_a] <= din_a;
        end
    end

    // Logic Read (Port B)
    always @(posedge clk) begin
        dout_b <= ram[addr_b];
    end

endmodule