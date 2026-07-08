`define DATA_WIDTH 16
`define IN_DIM 64  
`define OUT_DIM 128 
`define FRAC_BITS 12 

`define D_MODEL 64
`define SEQ_LEN 1000
`define D_STATE 16
`define D_CONV 4
`define EXPAND 2
`define D_INNER (`EXPAND * `D_MODEL)
`define DT_RANK ((`D_MODEL + 15) / 16)

// OP_MODE PE 
`define MODE_MAC 2'd0
`define MODE_MUL 2'd1
`define MODE_ADD 2'd2