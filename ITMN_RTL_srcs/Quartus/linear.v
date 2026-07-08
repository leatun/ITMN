/*

void linear(const model_dtype x[], model_dtype y[], 
            const model_dtype* W, const model_dtype b[],
            int in_dim, int out_dim) {
    
    for (int i = 0; i < out_dim; ++i) {
        model_dtype sum = 0.0f;
        for (int j = 0; j < in_dim; ++j) {
            sum += x[j] * W[i * in_dim + j];
        }
        y[i] = sum + (b ? b[i] : 0.0f);
    }
}

*/

`include "_parameter.v"

module linear
(
    input clk,
    input reset,
    input start,
    input signed [`DATA_WIDTH-1:0] data_in,
    output reg signed [`DATA_WIDTH-1:0] data_out,
    output reg done,
	 //Debug
	 output reg [3:0] state, next_state,
	 output reg [`DATA_WIDTH-1:0] i, j, k,
	 
	 output signed [`DATA_WIDTH-1:0] x_mem_out0,
	 output signed [`DATA_WIDTH-1:0] x_mem_out1,
	 output signed [`DATA_WIDTH-1:0] x_mem_out2,
	 
	 output reg [`DATA_WIDTH-1:0] temp,
	 
	 output reg signed [`DATA_WIDTH-1:0] x_reg,
	 output reg signed [`DATA_WIDTH-1:0] W_reg,
	 output reg signed [`DATA_WIDTH*2-1:0] sum_reg,
	 output wire signed [`DATA_WIDTH*2-1:0] mac_out
	 
);

    // --- Memories ---
    reg signed [`DATA_WIDTH-1:0] x_mem[`IN_DIM-1:0];
    reg signed [`DATA_WIDTH-1:0] W_mem[`OUT_DIM * `IN_DIM - 1 : 0];
    reg signed [`DATA_WIDTH-1:0] b_mem[`OUT_DIM-1:0];
    reg signed [`DATA_WIDTH-1:0] y_mem[`OUT_DIM-1:0];

    // --- FSM Registers ---
    //reg [3:0] state, next_state; // B? next_state, ch? c?n state
    //reg [`DATA_WIDTH-1:0] i, j, k; // Dï¿½ng reg thay cho integer ï¿½? d? debug
    
    //reg [`DATA_WIDTH-1:0] temp;
    
    // --- Datapath Registers ---
    //reg signed [`DATA_WIDTH-1:0] x_reg;
    //reg signed [`DATA_WIDTH-1:0] W_reg;
    //reg signed [`DATA_WIDTH*2-1:0] sum_reg;
    //wire signed [`DATA_WIDTH*2-1:0] mac_out;
    
    // PE
    mul_acc_wide pe0 (
        .clk(clk), .x(x_reg), .y(W_reg), .z(sum_reg), .s(mac_out)
    );

    // STATE register
    always @(posedge clk or posedge reset) begin
        if(reset) begin
            state <= `IDLE;
            i <= 0; j <= 0; k <= 0;
            sum_reg <= 0;
            data_out <= 0;
            done <= 0;
            temp <= 0;
        end else begin
            state <= next_state;
				
				case(state)

            `IDLE: begin
                done <= 0;
                i <= 0; j <= 0; k <= 0;
                temp <= 0;
                sum_reg <= 0;
            end

            `READ_X: begin
                x_mem[j] <= data_in;
                j <= (j == `IN_DIM - 1) ? 0 : j + 1;
            end

            `READ_W: begin
                W_mem[i] <= data_in;
                i <= (i == `OUT_DIM*`IN_DIM - 1) ? 0 : i + 1;
            end

            `READ_B: begin
                b_mem[i] <= data_in;
                i <= (i == `OUT_DIM - 1) ? 0 : i + 1;
                if(i == `OUT_DIM - 1) begin
                    j <= 0; k <= 0;
                    sum_reg <= 0;
                end
            end

            `COMPUTE: begin
                if(k == 0) begin
                    x_reg <= x_mem[j];
                    W_reg <= W_mem[i*`IN_DIM + j];
                    k <= 1;
                end else begin
                    sum_reg <= mac_out;
                    if(j == `IN_DIM - 1) begin
                        y_mem[i] <= mac_out[`DATA_WIDTH*2 - 1:`DATA_WIDTH] + b_mem[i];
                        sum_reg <= 0;
                        j <= 0;
                        i <= (i == `OUT_DIM - 1) ? 0 : i + 1;
                    end else begin
                        j <= j + 1;
                    end
                    k <= 0;
                end
            end

            `WRITE_Y: begin
                data_out <= y_mem[i];
                i <= (i == `OUT_DIM - 1) ? 0 : i + 1;
            end

            `DONE: begin
                done <= 1;
            end

        endcase
        end
    end
    
    // NEXT STATE + OUTPUT LOGIC
    always @(*) begin
        next_state = state;
        case(state)
            `IDLE:      if(start) next_state = `READ_X;
            `READ_X:    if(j == `IN_DIM - 1) next_state = `READ_W;
            `READ_W:    if(i == `OUT_DIM*`IN_DIM - 1) next_state = `READ_B;
            `READ_B:    if(i == `OUT_DIM - 1) next_state = `COMPUTE;
            `COMPUTE:   if(i == `OUT_DIM - 1 && j == `IN_DIM - 1 && k == 1) next_state = `WRITE_Y;
            `WRITE_Y:   if(i == `OUT_DIM - 1) next_state = `DONE;
            `DONE:      if(!start) next_state = `IDLE;
        endcase
    end
	 
	 assign x_mem_out0 = x_mem[0];
	 assign x_mem_out1 = x_mem[1];
	 assign x_mem_out2 = x_mem[2];
	 
    
endmodule