`include "_parameter.v"

module Linear_Layer
(
    input clk,
    input reset,

    input start,
    input [15:0] len, 
    input en,
    output reg done,

    input signed [`DATA_WIDTH-1:0] x_val,
    input signed [16 * `DATA_WIDTH - 1 : 0] W_row_vals,
    input signed [16 * `DATA_WIDTH - 1 : 0] bias_vals,

    output reg [16 * `DATA_WIDTH - 1 : 0] y_out,

    output reg [1:0] pe_op_mode_out,
    output reg       pe_clear_acc_out,

    output reg [16 * `DATA_WIDTH - 1 : 0] pe_in_a_vec,
    output reg [16 * `DATA_WIDTH - 1 : 0] pe_in_b_vec,

    input wire [16 * `DATA_WIDTH - 1 : 0] pe_result_vec
);

    reg [15:0] counter;      
    
    wire signed [`DATA_WIDTH-1:0] W_unpacked [15:0];
    wire signed [`DATA_WIDTH-1:0] bias_unpacked [15:0];

    // FSM 
    localparam S_IDLE      = 0;
    localparam S_CALC      = 1; 
    localparam S_WAIT_PE   = 2; 
    localparam S_ADD_BIAS  = 3; 
    localparam S_DONE      = 4;
    
    reg [2:0] state;

    // UNPACK DATA 
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : wiring
            assign W_unpacked[i]    = W_row_vals[i*`DATA_WIDTH +: `DATA_WIDTH];
            assign bias_unpacked[i] = bias_vals[i*`DATA_WIDTH +: `DATA_WIDTH];
        end
    endgenerate

    integer k;
    always @(*) begin
        pe_op_mode_out   = `MODE_MAC;
        pe_clear_acc_out = 0;
        pe_in_a_vec      = 0;
        pe_in_b_vec      = 0;

        case (state)
            S_IDLE: begin
                if (start) pe_clear_acc_out = 1;
                else pe_clear_acc_out = 0;
            end

            S_CALC: begin
                pe_op_mode_out   = `MODE_MAC;
                pe_clear_acc_out = 0;
                
                if (en) begin
                    for (k = 0; k < 16; k = k + 1) begin
                        pe_in_a_vec[k*16 +: 16] = x_val; 
                        pe_in_b_vec[k*16 +: 16] = W_unpacked[k];
                    end
                end else begin
                    // Khi pause (controller dang doc RAM): nap 0 de giu nguyen accumulator
                    // Acc = Acc + (0 * 0)
                    for (k = 0; k < 16; k = k + 1) begin
                        pe_in_a_vec[k*16 +: 16] = 0; 
                        pe_in_b_vec[k*16 +: 16] = 0;
                    end
                end
            end
            
            S_WAIT_PE: begin
                pe_op_mode_out   = `MODE_MAC;
                pe_clear_acc_out = 0;
                for (k = 0; k < 16; k = k + 1) begin
                    pe_in_a_vec[k*16 +: 16] = 0;
                    pe_in_b_vec[k*16 +: 16] = 0;
                end
            end

            S_ADD_BIAS: begin
                pe_op_mode_out   = `MODE_MAC;
                pe_clear_acc_out = 0;
                
                for (k = 0; k < 16; k = k + 1) begin
                    pe_in_a_vec[k*16 +: 16] = 16'd4096; // 1.0
                    pe_in_b_vec[k*16 +: 16] = bias_unpacked[k];
                end
            end
            
            S_DONE: begin
                pe_clear_acc_out = 0;
            end
        endcase
    end

    always @(*) begin
        // PE da tich luy va giu gia tri trong thanh ghi noi bo
        y_out = pe_result_vec;
    end

    // CONTROLLER 
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            counter <= 0;
            done <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        counter <= 0;
                        state <= S_CALC;
                    end
                end

                S_CALC: begin
                    if (en) begin
                        if (counter == len - 1) begin
                            state <= S_WAIT_PE; 
                        end else begin
                            counter <= counter + 1;
                        end
                    end
                end
                
                S_WAIT_PE: begin
                     state <= S_ADD_BIAS;
                end

                S_ADD_BIAS: begin
                    state <= S_DONE;
                end

                S_DONE: begin
                    done <= 1;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end
    

endmodule
