# ITMN_VIM - Vision Mamba (Vim) Adoption

**Date**: 2026-08-08
**Base HW**: ITMN_MARCA (Mamba1 stack, widened 9-bit ports)
**Target board**: VC709 (Virtex-7, matches FastMamba/LightMamba paper baseline)

## Vim vs standard Mamba1

Vim (Vision Mamba, Zhu et al. 2024) modifies Mamba1 for images:

1. **Patch embedding** front-end: image 224x224 -> 14x14 patches (16-pixel patch)
   -> 196 patch tokens + 1 CLS token = 197 sequence tokens.

2. **Bidirectional SSM per block**: each Vim block runs SSM in BOTH forward and
   backward direction, then merges. Standard Mamba1 is unidirectional.

   From Vim Alg 1:
       x_fwd = SSM(conv1d_fwd(SiLU(x)))         # forward direction
       x_bwd = SSM(reverse(conv1d_bwd(SiLU(reverse(x)))))  # backward
       y     = out_proj( (x_fwd + x_bwd) * SiLU(z) )       # merge via z-gate

3. **No dt_proj on token dim**: still uses Mamba1's dt_proj per-input.

## HW impl status (this folder)

- Current HW: **unidirectional Mamba1** (M6 scans forward only).
- Bidirectional cycle cost reported analytically in TB: `cyc_bidir = cyc_unidir + M6_cyc`.
- **Design tweak required for real HW**: add `SCAN_DIR` input to Mamba_Top,
  reverse M6 read/write addresses when SCAN_DIR=1, outer FSM runs M6 twice per
  block (fwd then bwd), merges outputs before M7.

## Vim configs (TBs in tb/)

| Model  | L  | D    | d_inner | N  | dt_rank | XP_OUT_GRP | Params | DMA/layer |
|--------|----|------|---------|----|---------|------------|--------|-----------|
| Vim-Ti | 24 | 192  | 384     | 16 | 12      | 3          | 7 M    | 16,184    |
| Vim-S  | 24 | 384  | 768     | 16 | 24      | 4          | 26 M   | 60,200    |
| Vim-B  | 24 | 768  | 1536    | 16 | 48      | 5          | 98 M   | 233,768   |

Sequence length for 224x224 image: **197 tokens** (196 patches + 1 CLS).

## Comparison target

- **Mamba-X** (KAIST ICCAD'25): cycle-level simulator, no real FPGA numbers
- **Vim paper** (Zhu 2024): GPU-only benchmark, no HW numbers
- No direct FPGA baseline in literature -> our numbers set a reference point

## Design tweak spec (future work)

If real HW bidirectional needed:

1. Add `input SCAN_DIR` to Mamba_Top (1'b0=fwd, 1'b1=bwd).
2. In M6 FSM, when SCAN_DIR=1:
   - Read address: `PT_U + (T_MAX - 1 - t_cnt)` instead of `PT_U + t_cnt`
   - Write address: `PT_Y_SSM + (T_MAX - 1 - t_cnt)` instead of `PT_Y_SSM + t_cnt`
3. Outer FSM state: BLOCK_START -> M6_FWD -> SAVE_YFWD -> H_RESET -> M6_BWD -> MERGE (y_fwd + y_bwd via add PE) -> M7 -> M8.
4. Cost: +M6_cyc per token, +1 URAM (or BRAM) for storing y_fwd during bwd pass.

Total effort: ~1-2 days RTL + verify.
