# Session 2026-05-20: Final Stage Investigation

## Status: ROOT CAUSE IDENTIFIED — awaiting golden regeneration

## Symptom
Block 0 simulation: all stages PASS except `Final Full Output` → 46306/64000 errors, max_d=30319.
All inception branches (Inc B1/B2/B3/B4) and all Mamba stages including Mam OutProj PASS with 0 errors.

## Debug method
Added `$display` in `ITM_CTRL_TB.v compare_all_stages` task (Final loop) to dump first 8 mismatches with raw inputs:
- per-c_grp error counters (`err_fin_c0..c3`)
- inception value, mamba value, BN scale, BN shift for each mismatch

## Debug output analysis (key samples)
```
[DBG #1] c=0 t=1 got=  9740 exp= 16712  inc=  -87 mam= 16712 scale= 75 shift= -1
[DBG #3] c=0 t=3 got= 19051 exp= 32767  inc= -250 mam= 32767 scale= 75 shift= -1
[DBG #5] c=0 t=5 got= 12392 exp= 21500  inc= -349 mam= 21500 scale= 75 shift= -1
[DBG] err per c_grp: c0=12134 c1=11126 c2=11407 c3=11639  (uniform → not address/bank bug)
```

Manual verification:
- **RTL `got` matches `bn_relu(sat_add(inc, mam), scale, shift)`** — the HW formula. RTL is CORRECT.
- **Golden `exp` matches `bn_relu(inc, scale, shift) + relu(mam)`** — the PyTorch formula.
  Example t=1: `bn_relu(-87, 75, -1) = 0`, `relu(16712) = 16712`, sum = 16712 = exp ✓

## Root cause
`golden_all/block_*/Final_ITM_Full_FP.txt` files are STALE from an earlier extractor run that used the PyTorch formula `relu(bn(inc)) + relu(mam)`. We later reverted `extract_itm_full.py` Phase 4 back to `bn_relu(inc+mam)` (matching RTL) but **did NOT regenerate the golden files**. So:
- RTL computes `bn_relu(inc+mam)` ✓
- Current extractor Phase 4 computes `bn_relu(inc+mam)` ✓
- Golden files on disk still contain `relu(bn(inc)) + relu(mam)` ✗

This is a workflow bug, not an RTL bug.

## Fix (single action)
```bash
cd ITMN_Pytorch
python extract_itm_full.py --out ../golden_all --all_blocks
```
Then re-run sim. All 5 blocks should pass Final.

## Pending cleanup (optional, after sim passes)
- Debug `$display` in `ITM_CTRL_TB.v` Final loop — can remove for clean log
- `m_we <= 0` adds in `S_M8_NEXT` (line 1245) and `S_FIN_NEXT` (line 1302) — defensive only, no functional effect; keep or revert as preferred

## Things RULED OUT during this session (do not re-investigate)
1. **Memory bank addressing**: `Memory_System.v` confirms `bank_sel=0 → read RAM_A / write RAM_B`; `bank_sel=1 → read RAM_B / write RAM_A`. Both `fin_branch_bank` and `S_FIN_WAIT2 bank_sel=0` are CORRECT for the current address layout.
2. **bn_relu / sat_add16 functions**: identical to extractor's `bn_relu_hw` / `sat_add` — verified line by line.
3. **BN scale/shift DMA loading**: `f_Inc_Scale[c_grp*16+i] → C_INC_SCALE+c_grp lane i` matches `scale_q[c_grp*16+i]`. Verified via debug dump (scale=75, shift=-1 for ch 0 matches PyTorch fold_bn output).
4. **Channel ordering**: `inc_cat_q = concat([b1,b2,b3,b4])` ↔ `fin_branch=0,1,2,3 → A_CH1, B_CH2, B_CH3, B_CH4`.
5. **m_we corruption hypothesis**: Originally suspected spurious writes from m_we staying 1 through Phase 4. Defensive fix added but error count unchanged → corruption was not actually happening (Vivado NBA semantics make spurious writes idempotent as long as m_wr_data is read in same edge it's updated).

---

# Block 4 Debug Summary

## Problem Statement
Block 4 simulation showed massive errors starting from Inception B3/B4 and cascading through all Mamba stages. Block 0-3 passed perfectly.

## Root Causes Found

### 1. **B4 Weight Loading Missing c_grp_br Loop** (CRITICAL)
**File:** `ITM_CTRL_TB.v` line 315-319

**Problem:** B4 weight loading code was missing the `c_grp_br` loop that B2/B3 had. This caused only the first output group (channels 0-15) to have valid weights, while the second group (channels 16-31) used garbage data.

**Fix:**
```verilog
// BEFORE (WRONG)
for (k = 0; k < 39; k = k + 1)
    for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
        ...
    end

// AFTER (CORRECT)
for (c_grp_br = 0; c_grp_br < (BLK_CH_OUT >= 8 ? 2 : 1); c_grp_br = c_grp_br + 1)
    for (k = 0; k < 39; k = k + 1)
        for (c = 0; c < BLK_CH_OUT*4; c = c + 1) begin
            for (i = 0; i < 16; i = i + 1)
                dma_wdata[i*16 +: 16] = f_Wb4[(c_grp_br*16+i)*(BLK_CH_OUT*4)*39 + c*39 + k];
            dma_write(2, DW_B4 + c_grp_br*39*(BLK_CH_OUT*4) + k*(BLK_CH_OUT*4) + c, dma_wdata);
        end
```

### 2. **compare_all_stages Parameter Naming Confusion** (CRITICAL)
**File:** `ITM_CTRL_TB.v` line 488-495

**Problem:** Task parameter was named `ch_in` but all callers passed `BLK_CH_OUT`. This caused:
- `dim_cmp = ch_in * 4` calculated as `4*4=16` instead of `8*4=32` for block 4
- `br_grps = (ch_in >= 8)` evaluated as `(4>=8)=false` instead of `(8>=8)=true`
- Only compared 16 channels (4000 values) instead of 32 channels (8000 values) per branch

**Fix:**
```verilog
// BEFORE
task compare_all_stages;
    input integer T;
    input integer ch_in;  // WRONG NAME
    ...
    dim_cmp = ch_in * 4;
    br_grps = (ch_in >= 8) ? 2 : 1;

// AFTER
task compare_all_stages;
    input integer T;
    input integer ch_out;  // CORRECT NAME
    ...
    dim_cmp = ch_out * 4;
    br_grps = (ch_out >= 8) ? 2 : 1;
```

### 3. **Display Message Bug** (Cosmetic)
**File:** `ITM_CTRL_TB.v` line 1036

**Problem:** Display used `BLK_CH_OUT*16` to show d_in, causing misleading log message "d_in=128" when actual d_in=64.

**Fix:**
```verilog
// BEFORE
$display("\n>> [BLOCK 4] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
         BLK_T, BLK_CH_OUT*16, BLK_D_INNER, BLK_DT_RANK);

// AFTER
$display("\n>> [BLOCK 4] T=%0d  d_in=%0d  d_inner=%0d  dt_rank=%0d",
         BLK_T, BLK_CH_IN*16, BLK_D_INNER, BLK_DT_RANK);
```

### 4. **Comment Correction** (Documentation)
**File:** `ITM_CONTROLLER.v` line 10

**Problem:** Comment incorrectly stated block 4 has `CH_IN=8 (d_in=128)`.

**Fix:**
```verilog
// BEFORE
//   block 4:   T=250,  CH_IN=8 (d_in=128),  CH_M=16 (d_inner=256), DT_RANK=8

// AFTER
//   block 4:   T=250,  CH_IN=4 (d_in=64), CH_OUT=8 (d_out=128), CH_M=16 (d_inner=256), DT_RANK=8
```

## Block 4 Architecture Verification

### PyTorch Model (ITMN.py line 147)
```python
ITMBlock(d_model, 2 * d_model)  # 64 → 128
```

### Dimensions
- **d_in:** 64 channels (CH_IN=4)
- **d_out:** 128 channels (CH_OUT=8)
- **d_inner:** 256 channels (CH_M=16)
- **dim:** 32 channels per inception branch (d_out/4)
- **T:** 250 timesteps

### Expected Sizes (T=250)
| Stage | Channels | Size | Formula |
|-------|----------|------|---------|
| P1 Output | 128 | 32000 | 128 × 250 |
| Inc Bot/B1/B2/B3/B4 | 32 each | 8000 each | 32 × 250 |
| Z_Gate | 256 | 64000 | 256 × 250 |
| U_Silu | 256 | 64000 | 256 × 250 |
| X_Proj | 48 (padded) | 12000 | 48 × 250 |
| Delta | 256 | 64000 | 256 × 250 |
| H_State | 256 | 4096 | 256 × 16 |
| Y_Gated | 256 | 64000 | 256 × 250 |
| OutProj | 128 | 32000 | 128 × 250 |
| Final | 128 | 32000 | 128 × 250 |

### Golden Files Verification
All golden files in `golden_all/block_04_layer06/` match expected sizes ✓

## Impact Analysis

### Before Fixes
- Inc B3/B4: Channels 16-31 had garbage weights → completely wrong outputs
- Testbench only compared channels 0-15, missing half the errors
- Mamba stages received corrupted inception output → cascade failures

### After Fixes
- B4 weights loaded correctly for all 32 channels
- Testbench compares all 32 channels per branch
- Expected: All stages should pass with err=0

## Files Modified
1. `ITMN_RTL_srcs/sources_1/new/ITM_CONTROLLER.v` (comment only)
2. `ITMN_RTL_srcs/sim_1/new/ITM_CTRL_TB.v` (3 fixes)

## Next Steps
Run simulation to verify all fixes resolve the block 4 errors.
