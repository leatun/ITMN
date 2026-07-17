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

// Weight RAM (BRAM target=2, depth W_MEM_DEPTH=8192, TDP for MAC2 dual read)
//   Layout sized for worst-case B4 (d_inner=256, dt_rank=8):
//     [0..1216)      SMALLS resident (B0..B4):
//                      W_DW      [0..64)     max = d_inner*4/16 = 64 (B4)
//                      W_XPROJ   [64..832)   max = 48*d_inner/16 = 768 (B4)
//                      W_DTPROJ  [832..960)  max = d_inner*dt_rank/16 = 128 (B4)
//                      W_A       [960..1216) max = d_inner*d_state/16 = 256 (B4)
//     [1216..3264)   Slot X   — W_INPROJ_X permanent (max 2048 words for B4)
//     [3264..5312)   Slot Z   — W_INPROJ_Z permanent (max 2048 words for B4)
//     [5312..7360)   Slot OUT — W_OUTPROJ permanent (max 2048 words for B4);
//                               streaming removed (2026-07-13 refactor) —
//                               all weights fit permanently, TDP + no runtime
//                               DMA writes → 2R/cyc always available.
//     [7360..8192)   spare
`define W_MEM_DEPTH       8192
`define W_MEM_ADDR_W      13         // ceil(log2(8192))
`define W_DW_BASE         15'd0
`define W_XPROJ_BASE      15'd64
`define W_DTPROJ_BASE     15'd832
`define W_A_BASE          15'd960
`define W_INPROJ_X_BASE   15'd1216
`define W_INPROJ_Z_BASE   15'd3264
`define W_OUTPROJ_BASE    15'd5312

// Const RAM (target=2)
`define C_W_NORM_BASE     15'd0
`define C_B_DW_BASE       15'd8
`define C_B_DT_BASE       15'd24
`define C_D_PARAM_BASE    15'd40

// ============================================================================
// Datapath geometry (added Phase 1)
// ============================================================================
`define LANE_MAX          4'd15      // 16 lanes − 1
`define LANE_W            4          // lane counter width
`define NUM_LANES         5'd16

// M2 depthwise-conv geometry
`define M2_TAP_COUNT      3'd4
`define M2_TAP_MAX        2'd3       // tap_cnt roll-over
`define M2_WEIGHT_STRIDE  15'd4      // 4 taps × 16b = 1 word per c_grp

// Address-chain constants
`define DT_SHIFT_BASE     9'd256     // MSB of 256-bit word for B/C barrel-shift

// PE pipeline depth (mac_last2 = len − PE_PIPE_DEPTH)
`define PE_PIPE_DEPTH     9'd2

// Fixed-point saturation output values (16-bit)
`define SAT16_MAX         16'sh7FFF
`define SAT16_MIN         16'sh8000  // = -32768 signed

// Activation-LUT latencies (documentation; not yet used in cycle math)
`define SILU_LAT          3'd2
`define SOFTPLUS_LAT      3'd2
`define EXP_LAT           3'd2
`define RSQRT_LAT         3'd1

// cur_stage enum (matches encoding used in Mamba_Top FSM dispatch)
`define STG_M1A           4'd0
`define STG_M1B           4'd1
`define STG_M2            4'd2
`define STG_M3            4'd3
`define STG_M4            4'd4
`define STG_M5            4'd5
`define STG_M6            4'd6
`define STG_M7            4'd7
`define STG_M8            4'd8
`define STG_RN            4'd9

// Block-0 reference dimensions (runtime values come from CH_OUT/CH_M/DT_RANK ports)
`define B0_D_MODEL        64
`define B0_D_INNER        128
`define B0_D_STATE        16
`define B0_DT_RANK        4
`define B0_N_PAD          48
`define B0_T_TOT          1000

`endif
