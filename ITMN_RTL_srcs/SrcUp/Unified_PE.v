`include "_parameter.v"

module Unified_PE
(
    input clk,
    input reset,
    
    input [1:0] op_mode,   // MAC, MUL, ADD
    input clear_acc,       

    input signed [`DATA_WIDTH-1:0] in_A,
    input signed [`DATA_WIDTH-1:0] in_B,

    output reg signed [`DATA_WIDTH-1:0] out_val
);

    // Saturation
    localparam signed [`DATA_WIDTH-1:0] MAX_POS = 16'sh7FFF; // +32767
    localparam signed [`DATA_WIDTH-1:0] MIN_NEG = 16'sh8000; // -32768
    
    // Ket qua nhan tho (32-bit)
    wire signed [2*`DATA_WIDTH-1:0] mult_raw;
    
    // Ket qua nhan da dich bit (dieu chinh fixed-point)
    // Dich phai 12 bit de quay ve ti le Q3.12
    wire signed [2*`DATA_WIDTH-1:0] mult_shifted;

    // Ket qua tinh toan tam thoi (truoc khi bao hoa)
    // Can 32 bit de chua an toan cac phep cong don ma khong bi tran ngay lap tuc
    reg signed [31:0] temp_result;

    // Thanh ghi tich luy (Accumulator Register)
    // Luu trang thai hien tai cua PE
    reg signed [`DATA_WIDTH-1:0] acc_reg;


    // LOGIC Combinational

    assign mult_raw = in_A * in_B;
    
    assign mult_shifted = mult_raw >>> `FRAC_BITS;


    // MUX
    always @(*) begin
        case (op_mode)
            `MODE_MAC: begin 
                // Out = Acc + (A * B)
                temp_result = acc_reg + mult_shifted;
            end
            
            `MODE_MUL: begin
                // Out = (A * B)
                temp_result = mult_shifted; 
            end
            
            `MODE_ADD: begin
                // Out = A + B
                temp_result = in_A + in_B;
            end
            
            default: temp_result = 0;
        endcase
    end


    // LOGIC bSequential
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            acc_reg <= 0;
            out_val <= 0;
        end else if (clear_acc) begin
            acc_reg <= 0;
            out_val <= 0;
        end else begin
            // Logic Saturation / Clamping
            
            if (temp_result > 32767) begin
                acc_reg <= MAX_POS;
            end else if (temp_result < -32768) begin
                acc_reg <= MIN_NEG;
            end else begin
                acc_reg <= temp_result[`DATA_WIDTH-1:0];
            end
            
            out_val <= (temp_result > 32767) ? MAX_POS : 
                       ((temp_result < -32768) ? MIN_NEG : temp_result[`DATA_WIDTH-1:0]);
        end
    end

endmodule
