`ifndef PARAMETER_V3
`define PARAMETER_V3

// Q4.11 fixed-point
`define DATA_W            16
`define DATA_WIDTH        16
`define ACC_W             40
`define FRAC_BITS         11

// Legacy PE_Array opcodes
`define MODE_MAC          2'd0
`define MODE_MUL          2'd1
`define MODE_ADD          2'd2

// Mamba_PE modes
`define MAMBA_PE_IDLE     3'd0
`define MAMBA_PE_MAC      3'd1
`define MAMBA_PE_MUL      3'd2
`define MAMBA_PE_ADD      3'd3
`define MAMBA_PE_SSM      3'd4
`define MAMBA_PE_MAC2     3'd5
`define MAMBA_PE_MUL2     3'd6

// Inception_PE modes (Phase B placeholder)
`define INC_PE_IDLE       2'd0
`define INC_PE_MAC        2'd1
`define INC_PE_MUL        2'd2
`define INC_PE_ADD        2'd3

// H_RegFile geometry (SSM state, 1 word per channel)
`define H_ADDR_W          9
`define H_DEPTH           256

// ============================================================================
// Per-timestep unified memory layout
//   Single ram_main (no bank_sel). Sizes cover B0..B4 (CH_OUT<=8, CH_M<=16).
//   Slot 0..3 = lifetime-aliased scratch (16 words each).
//   Slot CIRC = X_INNER_CIRC 4-tap circular (64 words).
//   Slot BULK = INPUT ↔ MAMBA_OUT aliased (T×CH_OUT words).
// ============================================================================
`define PT_X_NORM         15'd0       // Slot 0 (also X_PROJ, Y_GATED — disjoint lifetimes)
`define PT_X_PROJ         15'd0
`define PT_Y_GATED        15'd0
`define PT_X_CONV         15'd16      // Slot 1 (also U, Y_SSM)
`define PT_U              15'd16
`define PT_Y_SSM          15'd16
`define PT_DELTA          15'd32      // Slot 2
`define PT_Z_GATE         15'd48      // Slot 3 (hold M1B→M7)
`define PT_X_INNER_CIRC   15'd64      // Slot CIRC — 4×16=64 words, addr = (t%4)*CH_M + c_grp
`define PT_INPUT          15'd128     // Slot BULK — preload input, T×CH_OUT words
`define PT_MAMBA_OUT      15'd128     // Slot BULK — aliased with INPUT (safe: INPUT[t] dies before MAMBA_OUT[t] writes)
`define PT_MAIN_DEPTH     15'd4128    // 128 + 4000 (B0: T=1000, CH_OUT=4)

// Weight RAM (BRAM target=1)
`define W_INPROJ_X_BASE   15'd0
`define W_INPROJ_Z_BASE   15'd512
`define W_OUTPROJ_BASE    15'd1024
`define W_DW_BASE         15'd1536
`define W_XPROJ_BASE      15'd1600
`define W_DTPROJ_BASE     15'd2368
`define W_A_BASE          15'd2400

// Const RAM (target=2)
`define C_W_NORM_BASE     15'd0
`define C_B_DW_BASE       15'd8
`define C_B_DT_BASE       15'd24
`define C_D_PARAM_BASE    15'd40

// Block-0 reference dimensions (runtime values come from CH_OUT/CH_M/DT_RANK ports)
`define B0_D_MODEL        64
`define B0_D_INNER        128
`define B0_D_STATE        16
`define B0_DT_RANK        4
`define B0_N_PAD          48
`define B0_T_TOT          1000

`endif
