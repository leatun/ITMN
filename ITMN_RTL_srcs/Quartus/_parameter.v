`define DATA_WIDTH 16
`define IN_DIM 4   // 64
`define OUT_DIM 3  // 128

// Đ?nh ngh?a các tr?ng thái cho FSM
`define IDLE      4'b0000
`define READ_X    4'b0001
`define READ_W    4'b0010
`define READ_B    4'b0011
`define COMPUTE   4'b0100
`define WRITE_Y   4'b0101
`define DONE      4'b0110