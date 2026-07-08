`include "_parameter.v"

module Scan_Core_Engine
(
    input clk,
    input reset,
    
    input start,           
    input en,           
    input clear_h,         // Reset Hidden State
    output reg done,       
    
    // Inputs
    input signed [`DATA_WIDTH-1:0] delta_val,
    input signed [`DATA_WIDTH-1:0] x_val,
    input signed [`DATA_WIDTH-1:0] D_val,     
    input signed [`DATA_WIDTH-1:0] gate_val,  
    
    input signed [16 * `DATA_WIDTH - 1 : 0] A_vec,
    input signed [16 * `DATA_WIDTH - 1 : 0] B_vec,
    input signed [16 * `DATA_WIDTH - 1 : 0] C_vec,

    // Output
    output reg signed [`DATA_WIDTH-1:0] y_out,

    // ============================================================
    // PE ARRAY EXTERNAL
    // ============================================================

    output reg [1:0] pe_op_mode_out,
    output reg       pe_clear_acc_out,

    // Data dua vao pe (16 PE * 16 bit)
    output reg [16 * `DATA_WIDTH - 1 : 0] pe_in_a_vec,
    output reg [16 * `DATA_WIDTH - 1 : 0] pe_in_b_vec,

    input wire [16 * `DATA_WIDTH - 1 : 0] pe_result_vec
);

    // Internal Registers
    reg signed [`DATA_WIDTH-1:0] h_reg [15:0];
    reg signed [`DATA_WIDTH-1:0] discA_stored [15:0];   
    reg signed [`DATA_WIDTH-1:0] deltaBx_stored [15:0]; 
    
    // Internal Wires for Local Units
    wire signed [`DATA_WIDTH-1:0] A_in [15:0];
    wire signed [`DATA_WIDTH-1:0] B_in [15:0];
    wire signed [`DATA_WIDTH-1:0] C_in [15:0];
    
    // Exp Unit & SiLU Unit
    wire signed [`DATA_WIDTH-1:0] exp_in [15:0];
    wire signed [`DATA_WIDTH-1:0] exp_out [15:0];
    wire signed [`DATA_WIDTH-1:0] silu_out; 
    
    // Residual + Gating
    reg signed [31:0] Dx_prod;
    reg signed [31:0] y_with_D;
    reg signed [31:0] y_final_raw;

    // Unpack 
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : unpack
            assign A_in[i] = A_vec[i*`DATA_WIDTH +: `DATA_WIDTH];
            assign B_in[i] = B_vec[i*`DATA_WIDTH +: `DATA_WIDTH];
            assign C_in[i] = C_vec[i*`DATA_WIDTH +: `DATA_WIDTH];
            
            // Wiring Exp Unit
            // Exp Unit lay input tu ket qua PE tra ve (khi tinh xong Delta * A)
            assign exp_in[i] = pe_result_vec[i*`DATA_WIDTH +: `DATA_WIDTH];
            
            Exp_Unit exp_u (
                .clk(clk),
                .in_data(exp_in[i]),
                .out_data(exp_out[i])
            );
        end
    endgenerate

    SiLU_Unit_PWL silu_u (
        .clk(clk),
        .in_data(gate_val),
        .out_data(silu_out)
    );

    // Adder Tree (Combinational) - cong 16 ket qua tu PE (khi tinh C*h)
    reg signed [31:0] sum_all;
    integer k;
    always @(*) begin
        sum_all = 0;
        for (k=0; k<16; k=k+1) begin
            // PE result (h[i]*C[i])
            sum_all = sum_all + $signed(pe_result_vec[k*`DATA_WIDTH +: `DATA_WIDTH]);
        end
    end

    // FSM
    reg [3:0] state;
    localparam S_IDLE  = 0;
    localparam S_STEP1 = 1; // Calc Delta * A
    localparam S_STEP2 = 2; // Calc Delta * B
    localparam S_STEP3 = 3; // Calc (DeltaB) * x
    localparam S_STEP4 = 4; // Calc discA * h_old
    localparam S_STEP5 = 5; // Calc h_new = ... + ...
    localparam S_STEP6 = 6; // Calc C * h_new
    localparam S_STEP7 = 7; // Final Output

    integer j;

    // SEQUENTIAL LOGIC
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            done <= 0;
            y_out <= 0;
            for(j=0; j<16; j=j+1) begin
                h_reg[j] <= 0;
                discA_stored[j] <= 0;
                deltaBx_stored[j] <= 0;
            end
        end else begin
            if (clear_h) begin
                for(j=0; j<16; j=j+1) h_reg[j] <= 0;
            end
        
            if (start) begin
                state <= S_STEP1;
                done <= 0;
            end 
            else if (en) begin 
                case(state)
                    
                    S_STEP1: state <= S_STEP2;
                    
                    S_STEP2: state <= S_STEP3;
                    
                    S_STEP3: begin
                        for(j=0; j<16; j=j+1) discA_stored[j] <= exp_out[j];
                        state <= S_STEP4;
                    end
                    
                    S_STEP4: begin
                        for(j=0; j<16; j=j+1) deltaBx_stored[j] <= pe_result_vec[j*16 +: 16];
                        state <= S_STEP5;
                    end
                    
                    S_STEP5: state <= S_STEP6;
                    
                    S_STEP6: begin
                        for(j=0; j<16; j=j+1) h_reg[j] <= pe_result_vec[j*16 +: 16];
                        state <= S_STEP7;
                    end

                    S_STEP7: begin
                        
                        Dx_prod = x_val * D_val; 
                        y_with_D = sum_all + (Dx_prod >>> `FRAC_BITS);
                        y_final_raw = (y_with_D * silu_out) >>> `FRAC_BITS;
                        
                        if (y_final_raw > 32767) y_out <= 32767;
                        else if (y_final_raw < -32768) y_out <= -32768;
                        else y_out <= y_final_raw[15:0];
                        
                        done <= 1;
                        state <= S_IDLE;
                    end
                    
                    default: state <= S_IDLE;
                endcase
            end
            
            
            if (done && !start) done <= 0; 
        end
    end

    // Combinational Logic
    always @(*) begin
        pe_op_mode_out   = `MODE_MUL;
        pe_clear_acc_out = 0;
        pe_in_a_vec      = 0;
        pe_in_b_vec      = 0;

        case(state)
            S_STEP1: begin 
                pe_op_mode_out = `MODE_MUL;
                for(j=0; j<16; j=j+1) begin
                    pe_in_a_vec[j*16 +: 16] = delta_val;
                    pe_in_b_vec[j*16 +: 16] = A_in[j];
                end
            end

            S_STEP2: begin 
                pe_op_mode_out = `MODE_MUL;
                for(j=0; j<16; j=j+1) begin
                    pe_in_a_vec[j*16 +: 16] = delta_val;
                    pe_in_b_vec[j*16 +: 16] = B_in[j];
                end
            end

            S_STEP3: begin 
                pe_op_mode_out = `MODE_MUL;
                for(j=0; j<16; j=j+1) begin
                    pe_in_a_vec[j*16 +: 16] = pe_result_vec[j*16 +: 16]; 
                    pe_in_b_vec[j*16 +: 16] = x_val;
                end
            end

            S_STEP4: begin 
                pe_op_mode_out = `MODE_MUL;
                for(j=0; j<16; j=j+1) begin
                    pe_in_a_vec[j*16 +: 16] = discA_stored[j];
                    pe_in_b_vec[j*16 +: 16] = h_reg[j];
                end
            end

            S_STEP5: begin 
                pe_op_mode_out = `MODE_ADD;
                for(j=0; j<16; j=j+1) begin
                    pe_in_a_vec[j*16 +: 16] = pe_result_vec[j*16 +: 16];
                    pe_in_b_vec[j*16 +: 16] = deltaBx_stored[j];
                end
            end

            S_STEP6: begin 
                pe_op_mode_out = `MODE_MUL;
                for(j=0; j<16; j=j+1) begin
                    pe_in_a_vec[j*16 +: 16] = pe_result_vec[j*16 +: 16];
                    pe_in_b_vec[j*16 +: 16] = C_in[j];
                end
            end
            
            S_STEP7: begin
                pe_clear_acc_out = 0;
            end
        endcase
    end 
        


endmodule
