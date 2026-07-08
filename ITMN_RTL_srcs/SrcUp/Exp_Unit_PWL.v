`include "_parameter.v"

module Exp_Unit_PWL
(
    input clk,
    input signed [`DATA_WIDTH-1:0] in_data, // Q3.12
    output reg signed [`DATA_WIDTH-1:0] out_data
);

    // Distributed ROM: 64 segments x 32 bits
    (* rom_style = "distributed" *) reg [31:0] rom [0:63];

    initial begin
        $readmemh("exp_pwl_coeffs.mem", rom);
    end

    // Address Generation
    wire [5:0] addr;
    assign addr = in_data[15:10];

    // Fetch Coefficients
    wire signed [15:0] slope_comb;
    wire signed [15:0] intercept_comb;
    assign {slope_comb, intercept_comb} = rom[addr];
    
    // Calculation
    reg signed [31:0] prod;
    reg signed [31:0] res;
    
    // Constants
    localparam signed [31:0] MAX_VAL = 32'd32767;

    always @(posedge clk) begin
        // y = slope * x + intercept
        prod = slope_comb * in_data;
        res = (prod >>> `FRAC_BITS) + intercept_comb;
        
        // SATURATION 
        if (res > MAX_VAL) begin
            out_data <= MAX_VAL[15:0];      // Overflow (x > 2.08) -> 7.999
        end 
        else if (res < 0) begin
            out_data <= 16'd0;         
        end 
        else begin
            out_data <= res[15:0];      
        end
    end

endmodule