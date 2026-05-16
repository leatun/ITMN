`ifndef _BLOCK_PARAMS_V
`define _BLOCK_PARAMS_V

// ============================================================================
// _block_params.v - Per-block configuration for ITMN (5 ITM blocks)
//
// Model layout (from extract log + paper Table 1):
//   block_id | layers_idx | T_IN | T_OUT | d_in | d_inner | d_state | dt_rank
//       0    |      0     | 1000 |  1000 |  64  |   128   |   16    |   4
//       1    |      1     | 1000 |  1000 |  64  |   128   |   16    |   4
//    [MaxPool]            | 1000 |   500 |
//       2    |      3     |  500 |   500 |  64  |   128   |   16    |   4
//       3    |      4     |  500 |   500 |  64  |   128   |   16    |   4
//    [MaxPool]            |  500 |   250 |
//       4    |      6     |  250 |   250 | 128  |   256   |   16    |   8
//
// Weight RAM map (fixed, shared, reloaded per block):
//   All blocks reuse the same physical weight/const RAM.
//   Outer FSM (ITMN_TOP) DMAs in new weights before each block start.
//
// Data RAM usage per block (T=current block T):
//   RAM A (32K ? 256-bit):
//     0 .. T*4-1         Input X          (C_IN=64 ? T, 16ch/word ? T*4 words)
//     4000 .. 4000+T-1   BOT_out          (C_BOT=16 ? T, 16ch/word ? T words)
//     5000 .. 5000+T-1   B1_out           (16ch ? T ? T words)
//     8000 .. 8000+T*CH_M-1  Final c_grp 1..CH_M-1 (CH_M words per t group)
//     12000 .. 12000+T*CH_M*2-1  x_inner ? u_silu ? delta  (16ch?T ? 2*CH_M words)
//     MAMBA_OUT region   z_gate reuse after M7 ? mamba_out (M8)
//     H_STATE_BASE       h state (CH_M_GROUPS?16 words)
//
//   RAM B (32K ? 256-bit):
//     0 .. T*4-1         P1_out
//     4000 .. 4000+T-1   B2_out
//     5000 .. 5000+T-1   B3_out
//     6000 .. 6000+T-1   B4_out
//     8000 .. 8000+T-1   Final c_grp 0
//     12000 .. 12000+T*CH_M*2-1  x_conv ? x_proj_out
//     16000 .. 16000+T*CH_M*2-1  u_safe (M3_COPY)
//     24000 .. 24000+T*CH_M*2-1  y_ssm ? y_gated
//
// CH_M = d_inner/16, CH_IN = d_in/16, CH_OUT = d_in/16 (output = input channels)
//
// For block 0,1,2,3: CH_IN=4, CH_M=8, CH_OUT=4
// For block 4:       CH_IN=8, CH_M=16, CH_OUT=8
// ============================================================================

// Block ID constants (outer FSM uses these)
`define BLK_0  3'd0
`define BLK_1  3'd1
`define BLK_2  3'd2
`define BLK_3  3'd3
`define BLK_4  3'd4

// ============================================================================
// Per-block params as a ROM function (use in always block via case)
//
// Usage in RTL:
//   reg [9:0] blk_T;       // timesteps (1000, 500, 250)
//   reg [3:0] blk_CH_IN;   // d_in / 16
//   reg [3:0] blk_CH_M;    // d_inner / 16
//   reg [2:0] blk_DT_RANK; // dt_rank (4 or 8)
//   always @(*) begin
//       case (block_id)
//           3'd0, 3'd1: blk_T = 1000; ...
//           ...
//       endcase
//   end
// ============================================================================

// ============================================================================
// Derived RAM address constants as a function of block params.
// All addresses computed as functions of T and CH_M (= d_inner/16).
//
// RAM A layout:
`define A_INPUT_DEF       15'd0                    // C_IN groups x T words
`define A_BOT_OUT         15'd4000                 // 1 group ? T (BOT always 16ch)
`define A_B1_OUT          15'd5000                 // 1 group ? T
`define A_FINAL_OUT       15'd8000                 // (CH_M-1) groups ? T words each
`define A_X_INNER_DEF     (15'd12000)              // (2*CHM) groups x T  (x_inner, u, delta)
// z_gate/mamba_out: shared region after x_inner
// z_gate base = 12000 + 2*CHM*T
// mamba_out base = same (reuse after M7)
// h_state base: after z_gate+mamba_out (CHM*2*T words for z), + CHM*T for mamba_out
//             = 12000 + 2*CHM*T + CHM*T = 12000 + 3*CHM*T ... but mamba_out is CH_IN groups
// Simpler: fix h_state at high address (28000) to avoid overlap for all T

// RAM B layout:
`define B_P1_OUT          15'd0                    // CH_IN groups ? T
`define B_B2_OUT          15'd4000                 // 1 group ? T
`define B_B3_OUT          15'd5000                 // 1 group ? T
`define B_B4_OUT          15'd6000                 // 1 group ? T
`define B_FINAL_OUT       15'd8000                 // 1 group ? T (c_grp=0)
`define B_X_CONV          15'd12000                // CHM groups ? T (x_conv ? x_proj)
// u_safe: after x_conv region = 12000 + CHM*T
// For max CHM=16, T=1000: 12000+16000=28000. Need 28000+16*1000=44000 > 32K!
// Fix: for T=1000 CHM=8 ? u_safe=12000+8000=20000, size=8*1000=8000, top=28000 ?
//       T=500  CHM=8 ? u_safe=12000+4000=16000, size=8*500=4000, top=20000  ?
//       T=250  CHM=16 ? u_safe=12000+4000=16000, size=16*250=4000, top=20000 ?
//               x_conv = 16*250=4000 words ? 12000..15999, u_safe at 16000 ?
// y_ssm: after u_safe = u_safe_base + CHM*T
//         T=1000 CHM=8: 20000+8000=28000, size=8000, top=36000 > 32K! TOO BIG
// Need different layout for large T:
//   Option: y_ssm REUSES x_conv region after x_proj is done (M5 overwrites delta ? done with x_proj)
//   But M6b needs x_proj (for B,C) while writing y_ssm...
//   ? y_ssm at 24000 (as current): for T=1000 CHM=8: size=8*1000=8000, top=32000 ? (barely)
//   ? for T=250 CHM=16: size=16*250=4000, top=28000 ?

// Conclusion: current base addresses work for all 5 blocks.
// z_gate: 12000 + 2*CHM*T
//   T=1000 CHM=8 ? 12000+16000=28000, top=28000+8000=36000 > 32K !! OVERFLOW
//
// Problem: x_inner takes 12000..12000+CHM*2*T-1
//          For T=1000, CHM=8: 12000..27999 (16000 words) ? 28000 free
//          For T=500,  CHM=8: 12000..19999 (8000 words)
//          For T=250,  CHM=16: 12000..19999 (8000 words)
// z_gate at 28000 (fixed high):
//          T=1000 CHM=8: 28000..28000+8*1000-1=35999 > 32K !!
// 
// z_gate cannot go at 28000 for T=1000, CHM=8.
// 
// Solution: For T=1000, CHM=8:
//   Available RAM A after input(4000) + branches(5000..6999) + final(8000..11999)
//   and x_inner(12000..27999) ? only 28000..32767 free = 4767 words
//   z_gate needs 8*1000=8000 > available!
// 
// REDESIGN: x_inner/u/delta are chained overlays of SAME region.
//   z_gate is INDEPENDENT (written at M1b, read at M7, not overwritten in between).
//   Current V8e: x_inner occupies 12000..19999 (CH_M=8 ? T/2 ? 2 banks), z_gate at 20000..27999.
//   ? x_inner: 12000..12000+CHM*T-1 (CHM groups ? T words each)
//               T=1000 CHM=8: 12000..19999 (8000 words) ?
//               T=500  CHM=8: 12000..15999 (4000 words) ?
//               T=250  CHM=16: 12000..15999 (4000 words) ?
//   ? z_gate: 20000..20000+CHM*T-1
//               T=1000 CHM=8: 20000..27999 (8000 words) ?
//               T=500  CHM=8: 20000..23999 (4000 words) ?
//               T=250  CHM=16: 20000..23999 (4000 words) ?
//   ? mamba_out (reuses z_gate after M7): 20000..20000+CH_IN*T-1 [CH_IN=4 or 8]
//               T=1000 CH_IN=4: 20000..23999 ? (z_gate was ..27999, safe to reuse 20000..23999)
//               T=250  CH_IN=8: 20000..21999 ?
//   ? h_state: 28000..28000+CHM*16-1 (CHM groups ? 16 states)
//               T=1000 CHM=8:  28000..28127 (128 words) ?
//               T=250  CHM=16: 28000..28255 (256 words) ?
//
// RAM B:
//   ? x_conv: 12000..12000+CHM*T-1
//   ? u_safe: 20000..20000+CHM*T-1  [was 16000 in V8, move to 20000 for simplicity]
//             T=1000 CHM=8: 20000..27999 ? (8000 words)
//             T=500  CHM=8: 20000..23999 ?
//             T=250  CHM=16: 20000..23999 ?
//   ? y_ssm:  same as B_X_CONV (reuse after x_proj done in M5):
//             M6 starts reading x_proj (B,C) AND writing y_ssm to different location.
//             Safe if y_ssm ? x_conv: put y_ssm at 24000..24000+CHM*T-1
//             T=1000 CHM=8: 24000..31999 (8000 words) = exactly 32K ?
//             T=500  CHM=8: 24000..27999 ?
//             T=250  CHM=16: 24000..27999 ?
//
// FINAL MEMORY MAP (same for all blocks, addresses are fixed):
`define A_Z_GATE          15'd20000     // CHM*T words (T=1000,CHM=8?8000; T=250,CHM=16?4000)
`define A_MAMBA_OUT       15'd28128     // moved from 20000 to preserve Z_GATE
`define A_H_STATE         15'd28000     // CHM*16 words
`define B_U_SAFE          15'd15000     // CHM*T words - reuses stale x_conv region (after M3)
`define B_Y_SSM           15'd23000     // CHM*T words - no overlap with U_SAFE

// Weight RAM - same base addresses for all blocks (reloaded by outer FSM per block):
// P1:   0   .. P1_W_WORDS-1   (CH_OUT*CH_IN*16*16 = 64*64=4096 = 256 words)
// BOT:  256 .. 319            (16*64 = 1024 = 64 words)
// B1:   320 .. 383            (16*64 = 64 words)
// B2:   384 .. 527            (16*16*9 = 144 words, rounded)
// B3:   528 .. 831            (16*16*19 = 304 words)
// B4:   832 .. 1455           (16*16*39 = 624 words, rounded)
// M_X:  1456 .. 1967          (d_inner*d_in = 128*64=8192 = 512 words)
// M_Z:  1968 .. 2479          (same)
// M_DW: 2480 .. 2511          (d_inner*4/16 = 128/4 = 32 words)
// XPROJ:2512 .. 2895          (3 or 6 out_grps ? d_inner ? 384 words for CHM=8 or 768 for CHM=16)
// DTPROJ:2896 ..              (CHM?dt_rank/16 = 8?4/16 = 2 words OR 16?8/16=8 words)
// A_LOG: W_A_LOG_BASE         (CHM ? d_state = 128 or 256 words)
// D:    W_D_PARAM_BASE        (CHM words)
// OUTPROJ: W_OUTPROJ_BASE     (CH_OUT?d_inner = 4?128=512 or 8?256=2048 words)
//
// NOTE: For block 4 (d_inner=256 d_in=128): weights are larger.
//   XPROJ: ceil(40/16)=3 groups ? 256 in = 768 words. Starts at 2512, ends at 2512+768-1=3279.
//   Previously DTPROJ at 2896 - now starts at 3280. NEED TO SHIFT or use larger base.
//   See _block_params.v for exact per-block weight addresses.
`define W_P1_BASE         15'd0
`define W_BOT_BASE        15'd256
`define W_B1_BASE         15'd320
`define W_B2_BASE         15'd384
`define W_B3_BASE         15'd528
`define W_B4_BASE         15'd832
`define W_M_X_BASE        15'd1456
`define W_M_Z_BASE        15'd1968
`define W_M_DW_BASE       15'd2480
// Variable from here (depend on d_inner / dt_rank):
// For CHM=8 (d_inner=128): xproj_words = 3*128=384, so DTPROJ at 2512+384=2896 ?
// For CHM=16 (d_inner=256): xproj_words = 3*256=768, so DTPROJ at 2512+768=3280
// Strategy: DTPROJ_BASE = 2512 + xproj_words, calculated in ITM_CONTROLLER
`define W_XPROJ_BASE      15'd2512
// DTPROJ, A_LOG, D_PARAM, OUTPROJ: computed as XPROJ_BASE + xproj_words + offset

`endif