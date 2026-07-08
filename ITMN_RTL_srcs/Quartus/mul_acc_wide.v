// C?n t?o module PE m?i có đ?u ra r?ng hơn
module mul_acc_wide (
    input clk,
    input signed [`DATA_WIDTH-1:0] x,
    input signed [`DATA_WIDTH-1:0] y,
    input signed [`DATA_WIDTH*2-1:0] z,
    output reg signed [`DATA_WIDTH*2-1:0] s
);
    always @(posedge clk) begin
        s <= x * y + z;
    end
endmodule
