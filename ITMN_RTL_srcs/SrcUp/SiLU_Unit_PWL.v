`include "_parameter.v"

module SiLU_Unit_PWL
(
    input clk,
    input signed [`DATA_WIDTH-1:0] in_data, // Q3.12
    output reg signed [`DATA_WIDTH-1:0] out_data
);

    (* rom_style = "distributed" *) reg [31:0] rom [0:63];

    initial begin
        $readmemh("silu_pwl_coeffs.mem", rom);
    end

    // Address Generation
    wire [5:0] addr;
    assign addr = in_data[15:10];

    // Fetch Coefficients 
    reg signed [15:0] slope;
    reg signed [15:0] intercept;
    
    reg signed [15:0] in_data_d1;

    always @(posedge clk) begin
        //  ROM
        {slope, intercept} <= rom[addr];
        
        // Delay input
        in_data_d1 <= in_data;
    end
    
    wire signed [15:0] slope_comb;
    wire signed [15:0] intercept_comb;
    
    assign {slope_comb, intercept_comb} = rom[addr];
    
    
    reg signed [31:0] prod;
    reg signed [31:0] res;
    
    always @(posedge clk) begin
        
        // y = ax + b
        // slope (Q3.12) * input (Q3.12) -> Q6.24 -> Shift 12 -> Q3.12
        prod = slope_comb * in_data;
        
        // intercept (Q3.12)
        res = (prod >>> `FRAC_BITS) + intercept_comb;
        
        // Saturation
        if (res > 32767) out_data <= 32767;
        else if (res < -32768) out_data <= -32768;
        else out_data <= res[15:0];
    end

endmodule