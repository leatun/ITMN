`include "_parameter.v"

module Softplus_Unit_PWL
(
    input clk,
    input signed [`DATA_WIDTH-1:0] in_data, // Q3.12
    output reg signed [`DATA_WIDTH-1:0] out_data
);

    (* rom_style = "distributed" *) reg [31:0] rom [0:63];

    initial begin
        $readmemh("softplus_pwl_coeffs.mem", rom);
    end

    // Generation
    wire [5:0] addr;
    assign addr = in_data[15:10];

    // Fetch Coefficients
    wire signed [15:0] slope_comb;
    wire signed [15:0] intercept_comb;
    
    assign {slope_comb, intercept_comb} = rom[addr];
    
    // Calculation Logic
    reg signed [31:0] prod;
    reg signed [31:0] res;
    
    always @(posedge clk) begin
        // y = ax + b
        // slope (Q3.12) * input (Q3.12) -> prod (Q6.24)
        prod = slope_comb * in_data;
        
        res = (prod >>> `FRAC_BITS) + intercept_comb;
        
        if (res > 32767) out_data <= 32767;
        else if (res < -32768) out_data <= -32768;
        else out_data <= res[15:0];
    end

endmodule