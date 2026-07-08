`include "_parameter.v"

module Conv1D_Layer
(
    input clk,
    input reset,
    
    // Control
    input start,          
    input en,             
    input valid_in,      
    output reg valid_out, 
    output reg ready_in,  

    // Data
    input signed [16 * `DATA_WIDTH - 1 : 0] x_in_vec,
    input signed [16 * 4 * `DATA_WIDTH - 1 : 0] weights_vec,
    input signed [16 * `DATA_WIDTH - 1 : 0] bias_vec,

    // Output
    output signed [16 * `DATA_WIDTH - 1 : 0] y_out_vec,

    // External PE Interface 
    output reg [1:0] pe_op_mode_out,
    output reg       pe_clear_out,
    output reg [16 * `DATA_WIDTH - 1 : 0] pe_in_a_vec,
    output reg [16 * `DATA_WIDTH - 1 : 0] pe_in_b_vec,
    input wire [16 * `DATA_WIDTH - 1 : 0] pe_result_vec
);

    // Unpack
    wire signed [`DATA_WIDTH-1:0] x_in [15:0];
    wire signed [`DATA_WIDTH-1:0] w_in [15:0][3:0];
    wire signed [`DATA_WIDTH-1:0] b_in [15:0];

    genvar i, k;
    generate
        for (i = 0; i < 16; i = i + 1) begin : unpack
            assign x_in[i] = x_in_vec[i*`DATA_WIDTH +: `DATA_WIDTH];
            assign b_in[i] = bias_vec[i*`DATA_WIDTH +: `DATA_WIDTH];
            for (k = 0; k < 4; k = k + 1) begin : unpack_w
                assign w_in[i][k] = weights_vec[(i*4 + k)*`DATA_WIDTH +: `DATA_WIDTH];
            end
        end
    endgenerate

    // Shift Registers
    reg signed [`DATA_WIDTH-1:0] shift_reg [15:0][2:0]; 
    reg signed [`DATA_WIDTH-1:0] current_x [15:0]; 
    wire signed [`DATA_WIDTH-1:0] silu_out [15:0];
    
    // FSM
    reg [2:0] state; 
    localparam S_IDLE      = 0;
    localparam S_LOAD_BIAS = 1;
    localparam S_MAC_0     = 2;
    localparam S_MAC_1     = 3;
    localparam S_MAC_2     = 4;
    localparam S_MAC_3     = 5;
    localparam S_WAIT_SILU = 6; 
    localparam S_UPDATE    = 7;

    localparam signed [`DATA_WIDTH-1:0] ONE_FIXED = 16'h1000;

    // SiLU
    generate
        for (i = 0; i < 16; i = i + 1) begin : silu_gen
            SiLU_Unit u_silu (
                .clk(clk),
                .in_data(pe_result_vec[i*`DATA_WIDTH +: `DATA_WIDTH]), 
                .out_data(silu_out[i])
            );
            assign y_out_vec[i*`DATA_WIDTH +: `DATA_WIDTH] = silu_out[i];
        end
    endgenerate

    //LOGIC PE CONTROL (COMBINATIONAL)
    integer c;
    always @(*) begin
        // Default
        pe_op_mode_out = `MODE_MUL;
        pe_clear_out   = 0;
        pe_in_a_vec    = 0;
        pe_in_b_vec    = 0;

        case (state)
            S_IDLE: begin
                pe_clear_out = 1;
            end

            default: begin
                if (en) begin
                    case(state)
                        S_LOAD_BIAS: begin
                            pe_op_mode_out = `MODE_MUL; 
                            for (c=0; c<16; c=c+1) begin
                                pe_in_a_vec[c*16+:16] = b_in[c];
                                pe_in_b_vec[c*16+:16] = ONE_FIXED;
                            end
                        end
                        S_MAC_0: begin
                            pe_op_mode_out = `MODE_MAC;
                            for (c=0; c<16; c=c+1) begin
                                pe_in_a_vec[c*16+:16] = current_x[c]; 
                                pe_in_b_vec[c*16+:16] = w_in[c][0];
                            end
                        end
                        S_MAC_1: begin
                            pe_op_mode_out = `MODE_MAC;
                            for (c=0; c<16; c=c+1) begin
                                pe_in_a_vec[c*16+:16] = shift_reg[c][0]; 
                                pe_in_b_vec[c*16+:16] = w_in[c][1];
                            end
                        end
                        S_MAC_2: begin
                            pe_op_mode_out = `MODE_MAC;
                            for (c=0; c<16; c=c+1) begin
                                pe_in_a_vec[c*16+:16] = shift_reg[c][1]; 
                                pe_in_b_vec[c*16+:16] = w_in[c][2];
                            end
                        end
                        S_MAC_3: begin
                            pe_op_mode_out = `MODE_MAC;
                            for (c=0; c<16; c=c+1) begin
                                pe_in_a_vec[c*16+:16] = shift_reg[c][2]; 
                                pe_in_b_vec[c*16+:16] = w_in[c][3];
                            end
                        end
                        S_WAIT_SILU, S_UPDATE: begin
                            pe_op_mode_out = `MODE_MAC;
                            pe_clear_out = 0;
                        end
                    endcase
                end 
                else begin 
                    // (en=0): Keep Accumulator
                    pe_op_mode_out = `MODE_MAC;
                    pe_clear_out = 0;
                    // Input = 0
                end
            end
        endcase
    end


    // FSM (SEQUENTIAL) 
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            valid_out <= 0;
            ready_in <= 0;
            for (c = 0; c < 16; c = c + 1) begin
                shift_reg[c][0] <= 0;
                shift_reg[c][1] <= 0;
                shift_reg[c][2] <= 0;
                current_x[c] <= 0;
            end
        end else begin
            if (start) begin
                state <= S_IDLE;
                valid_out <= 0;
                ready_in <= 1;
                // Reset Shift Reg
                for (c = 0; c < 16; c = c + 1) begin
                    shift_reg[c][0] <= 0;
                    shift_reg[c][1] <= 0;
                    shift_reg[c][2] <= 0;
                end
            end 
            else begin
                case (state)
                    S_IDLE: begin
                        valid_out <= 0;
                        ready_in <= 1; 
                        
                        if (valid_in && en) begin
                            ready_in <= 0; // B?n r?n
                            for (c = 0; c < 16; c = c + 1)
                                current_x[c] <= x_in[c];
                            state <= S_LOAD_BIAS;
                        end
                    end
                    
                    default: begin
                        if (en) begin
                            case (state)
                                S_LOAD_BIAS: state <= S_MAC_0;
                                S_MAC_0:     state <= S_MAC_1;
                                S_MAC_1:     state <= S_MAC_2;
                                S_MAC_2:     state <= S_MAC_3;
                                S_MAC_3:     state <= S_WAIT_SILU;
                                S_WAIT_SILU: state <= S_UPDATE;
                                S_UPDATE: begin
                                    valid_out <= 1; 
                                    // Update Shift Reg
                                    for (c = 0; c < 16; c = c + 1) begin
                                        shift_reg[c][0] <= current_x[c];
                                        shift_reg[c][1] <= shift_reg[c][0];
                                        shift_reg[c][2] <= shift_reg[c][1];
                                    end
                                    state <= S_IDLE;
                                end
                            endcase
                        end
                    end
                endcase
            end
        end
    end

endmodule