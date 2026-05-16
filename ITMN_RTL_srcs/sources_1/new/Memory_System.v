`include "_parameter.v"

module Memory_System (
    input clk, input reset,
    input bank_sel,
    input  [14:0] core_read_addr,  output [255:0] core_read_data,
    input         core_write_en,   input [14:0] core_write_addr, input [255:0] core_write_data,
    input  [14:0] weight_read_addr, output [255:0] weight_read_data,
    input  [14:0] const_read_addr,  output [255:0] const_read_data,
    input         dma_write_en,
    input  [1:0]  dma_target,
    input  [14:0] dma_addr,
    input  [255:0] dma_wdata
);
    wire [255:0] out_ram_a, out_ram_b;
    
    wire we_a = (dma_write_en && dma_target==2'd0) || (core_write_en && bank_sel==1 && !dma_write_en);
    wire we_b = (dma_write_en && dma_target==2'd1) || (core_write_en && bank_sel==0 && !dma_write_en);
    wire we_w = (dma_write_en && dma_target==2'd2);
    wire we_c = (dma_write_en && dma_target==2'd3);

    wire [14:0]  addr_a_wr = (dma_write_en && dma_target==0) ? dma_addr : core_write_addr;
    wire [255:0] din_a     = (dma_write_en && dma_target==0) ? dma_wdata : core_write_data;
    wire [14:0]  addr_b_wr = (dma_write_en && dma_target==1) ? dma_addr : core_write_addr;
    wire [255:0] din_b     = (dma_write_en && dma_target==1) ? dma_wdata : core_write_data;

    BRAM_256b ram_a (.clk(clk), .we_a(we_a), .addr_a(addr_a_wr), .din_a(din_a),
                     .addr_b(bank_sel==0 ? core_read_addr : 15'd0), .dout_b(out_ram_a));
    BRAM_256b ram_b (.clk(clk), .we_a(we_b), .addr_a(addr_b_wr), .din_a(din_b),
                     .addr_b(bank_sel==1 ? core_read_addr : 15'd0), .dout_b(out_ram_b));
    BRAM_256b ram_weight (.clk(clk), .we_a(we_w), .addr_a(dma_addr), .din_a(dma_wdata),
                          .addr_b(weight_read_addr), .dout_b(weight_read_data));
    BRAM_256b ram_const  (.clk(clk), .we_a(we_c), .addr_a(dma_addr), .din_a(dma_wdata),
                          .addr_b(const_read_addr), .dout_b(const_read_data));

    assign core_read_data = (bank_sel == 0) ? out_ram_a : out_ram_b;
endmodule