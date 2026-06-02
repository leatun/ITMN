# Fmax / Resource Utilization Plan — ITM_Controller RTL

---

## 1. Verified baseline (before any Fmax work)

- Byte-exact sanity: **ALL PASS** (verify_byte_exact.py — P1_FP, Mamba, Final for all 5 blocks)
- AUC (hw, fb11_py): 0.9328, gap −0.0026 vs float
- Formula in RTL, extract, test_hw: `sat_add16(bn_relu(incep), relu16(mamba))`

---

## 2. Critical paths (combinational logic depth per clock cycle)

### CP-1 — `x_norm_fn` (M1A / M1B MAC path)

**Location:** `ITM_CONTROLLER.v:1567`, called in `S_M1A_MAC` and `S_M1B_MAC` substep 2  
**Logic depth:**
```
m_rd_data[i*16+:16] →
  p1_wide = x * gamma          # 16×16 → 32-bit signed
  p1_wide >>>= FRAC_BITS(11)
  p1      = sat16(p1_wide)
  out_wide = p1 * S            # 16×16 → 32-bit signed (S = norm_S_reg)
  out_wide >>>= FRAC_BITS(11)
  x_norm_fn = sat16(out_wide)
→ pe_A (drives PE accumulator)
```
Two 16-bit signed multiplies are **cascaded in a single combinational path**.  
The second multiply feeds from the saturation output of the first.  
The result drives `pe_A`, which clocks into the PE on the next edge.

**Proposed fix:** Split `x_norm_fn` into two registered stages.
- Cycle A (new substep): compute `p1 = sat16((x * gamma) >>> FB)`, register in a 256-bit intermediate `p1_norm_reg`
- Cycle B (current substep 2): compute `sat16((p1 * S) >>> FB)` from `p1_norm_reg`, load into `pe_A`

This removes one multiply from the critical path. The inner `S_M1A_MAC` / `S_M1B_MAC` substep loop adds one extra cycle per d_out word (d_out = 64 or 128), but the deeper path is removed.

---

### CP-2 — `norm_sq16_fn` (S_NORM_M1A_SQ_LATCH / S_NORM_M1B_SQ_LATCH)

**Location:** `ITM_CONTROLLER.v:1550`, assigned in `S_NORM_M1A_SQ_LATCH` and `S_NORM_M1B_SQ_LATCH`  
**Logic depth:**
```
norm_sq_acc <= norm_sq_acc + norm_sq16_fn(m_rd_data)

norm_sq16_fn:
  for j in 0..15:
    lane[j] = m_rd_data[j*16+:16]   # 16 lanes
    sq[j]   = lane[j] * lane[j]     # 16× signed 16-bit square → 32-bit
  acc = sum(sq[0..15])               # 16-to-1 adder tree, 40-bit wide
→ norm_sq_acc += acc                 # 40-bit add
```
A 16-input, 40-bit-wide adder tree in one clock cycle, preceded by 16 parallel squares.  
Adder tree depth: 4 levels (16→8→4→2→1), each level 40-bit wide.  
This is the most likely Fmax bottleneck.

**Proposed fix (Option A — 4-way pipelined accumulation):**
Replace the single SQ_LATCH state with 4 substeps, each processing 4 lanes (channels j*4 to j*4+3):
```
substate 0: partial[0] = sq(lanes 0-3)   # 4-to-1 tree, 35-bit
substate 1: partial[1] = sq(lanes 4-7)   # same
substate 2: partial[2] = sq(lanes 8-11)  # same
substate 3: norm_sq_acc += partial[0]+partial[1]+partial[2]+sq(lanes 12-15)
```
Each substate's adder tree is only 2 levels deep (4→2→1) instead of 4.  
Cost: 3 extra cycles per CH_OUT group × CH_OUT groups × T. For blocks 0-3: 3×4×1000 = 12K extra cycles. Acceptable.

**Proposed fix (Option B — Balanced binary tree + 1 register)**  
Same `norm_sq16_fn` content but split into two registered half-sums:
- Cycle 1: `half_a = sum(sq[0..7])` → registered
- Cycle 2: `norm_sq_acc += half_a + sum(sq[8..15])`  

Requires 2 substates instead of 1 in SQ_LATCH. Half as many levels per cycle. Simpler than Option A.

---

### CP-3 — `S_FIN_WRITE` (16 lanes bn_relu + relu16 + sat_add16)

**Location:** `ITM_CONTROLLER.v:1373–1394`  
**Logic depth per lane:**
```
incep_reg[i*16+:16]  →
  mul_raw  = raw * scale         # 16×16 → 32-bit
  bn_out   = sat16(mul_raw >>> 11)
  s        = bn_out + shift      # 17-bit add
  bn_out2  = sat16(s)
  x1       = relu16(bn_out2)
m_rd_data[i*16+:16]  →
  x2       = relu16(m_rd_data)
→ sat_add16(x1, x2)              # 17-bit add + sat
→ m_wr_data[i*16+:16]
```
One 16-bit multiply + 3 saturations + 2 adds, in parallel for 16 lanes.  
Likely 4–5 ns at 250 MHz (one DSP + LUT chain). Less deep than CP-1/CP-2 but has 16 copies.

**Proposed fix:** Move the multiply into the existing `S_FIN_MUL` state.  
Currently `S_FIN_MUL` only latches `incep_reg` and reads the shift address.  
With the fix:
- `S_FIN_MUL`: compute `mul_reg[i] <= $signed(incep_reg[i*16+:16]) * $signed(m_wr_data[i*16+:16])` for all 16 lanes (scale is in `m_wr_data` at that point). Register `mul_reg` (16×32-bit).
- `S_FIN_WRITE`: compute `bn_out = sat16(mul_reg[i] >>> FB)`, +shift, sat, relu, sat_add16 only.

This removes the multiply from `S_FIN_WRITE`'s critical path.  
`mul_reg` requires 16×32 = 512 bits of flip-flops — reasonable.

---

### CP-4 — `always @(*)` address stride multiplications

**Location:** `ITM_CONTROLLER.v:87–92`  
**Logic depth:**
```
always @(*) begin
  t_stride_in  = {5'd0, t_cnt} * {11'd0, CH_IN};    // 10×4-bit → 14-bit
  t_stride_m   = {5'd0, t_cnt} * {7'd0, ch_m_actual}; // 10×8-bit → 15-bit
  t_stride_out = {5'd0, t_cnt} * {11'd0, CH_OUT};   // 10×4-bit → 14-bit
  t_stride_xp  = {5'd0, t_cnt} * 15'd3;             // 10×2-bit → 12-bit
end
```
Combinational multiplies re-evaluated every cycle.  
These appear in address computations like `B_P1_OUT + t_stride_out + offset`.

**Proposed fix:** Replace with registered increment (adder, not multiplier).  
```verilog
always @(posedge clk) begin
    if (t_cnt == 0) begin
        t_stride_in  <= 0;
        t_stride_m   <= 0;
        t_stride_out <= 0;
        t_stride_xp  <= 0;
    end else if (t_cnt_inc) begin
        t_stride_in  <= t_stride_in  + {11'd0, CH_IN};
        t_stride_m   <= t_stride_m   + {7'd0,  ch_m_actual};
        t_stride_out <= t_stride_out + {11'd0, CH_OUT};
        t_stride_xp  <= t_stride_xp  + 15'd3;
    end
end
```
`t_cnt_inc` is a 1-cycle pulse when `t_cnt` increments (easy to derive from `t_cnt != t_cnt_prev`).  
This replaces 4 small multipliers with 4 adders on the critical path.  
Risk: timing if `t_cnt_inc` fires the cycle before the stride is first needed — needs a 1-cycle look-ahead or pre-increment at T_NEXT transitions.

---

## 3. RAM Reuse Opportunities

### RU-1 — Eliminate M3CP stage (alias u at A_X_INNER)

**Current flow:**
1. M3 (SiLU): reads `B_X_CONV`, writes SiLU(x_conv) → `A_X_INNER`
2. **M3CP (copy):** reads `A_X_INNER`, writes → `B_U_SAFE` (15000)
3. M4 (x_proj): reads `A_X_INNER`, writes → `B_X_CONV` (overwrites conv1d output with x_proj)
4. M6A_T2_READ, M6B_DU_READ: read `B_U_SAFE` (u = SiLU output, needed for dB×u and D×u)

**Key observation:** M4 does NOT write to `A_X_INNER`. `A_X_INNER` still holds SiLU output (u) throughout M4–M6. M3CP copies it to `B_U_SAFE` only because M6A/M6B happen to read from bank B.

**Fix:** Change `S_M6A_T2_READ` and `S_M6B_DU_READ` to read from `A_X_INNER` (bank A) instead of `B_U_SAFE` (bank B). Then remove the M3CP states entirely.

**Savings (approx):**  
M3CP has 5 FSM states per (c_grp_m, t): `READ, WAIT, LATCH, WRITE, NEXT`

| Block | CH_M | T | M3CP cycles saved |
|-------|------|---|-------------------|
| 0, 1  | 8    | 1000 | 5 × 8 × 1000 = **40,000** |
| 2, 3  | 8    | 500  | 5 × 8 × 500  = **20,000** |
| 4     | 16   | 250  | 5 × 16 × 250 = **20,000** |

Per-block savings of 20K–40K cycles. Per full inference (5 blocks): ~140K cycles.

**Implementation notes:**
- Set `bank_sel <= 0` in `S_M6A_T2_READ` and `S_M6B_DU_READ` (vs current `bank_sel <= 1`)
- Change `m_rd_addr` to `A_X_INNER + t_stride_m + {11'd0, c_grp_m}` in both states
- Remove S_M3CP_READ/WAIT/LATCH/WRITE/NEXT states and their transitions
- Change S_M3_NEXT: when done, transition to `S_M4_MAC` instead of `S_M3CP_READ`

---

### RU-2 — Fix A_MAMBA_OUT / A_H_STATE overlap for block 4

**Current addresses:**
```
A_H_STATE   = 28000   # d_inner × d_state / 16 words
A_MAMBA_OUT = 28128   # constant, was set when d_inner=128 → size=128 words
```

**Overlap for block 4** (CH_M=16, d_inner=256, d_state=16):
- A_H_STATE size = 256 × 16 / 16 = **256 words** → region [28000, 28256)
- A_MAMBA_OUT = 28128, which is **inside** A_H_STATE!
- M8 writes mamba_out to [28128..30128), overwriting the upper 128 words of H_STATE

**Why it works functionally:** By the time M8 runs, M6B has finished and H_STATE is not needed. No correctness impact, but it's a semantic hazard.

**Fix:** Make A_MAMBA_OUT a derived address: `A_MAMBA_OUT = A_H_STATE + max_h_state_words`

```verilog
// Parameterized: A_MAMBA_OUT starts right after H_STATE for any block config
wire [14:0] A_MAMBA_OUT = A_H_STATE + ({9'd0, ch_m_actual} * 15'd16 >> 4);
//  blocks 0-3: 28000 + 128 = 28128  (same as hardcoded today)
//  block 4:    28000 + 256 = 28256  (fixes the overlap)
```

Max total usage: 28256 + 2000 (block 4 mamba_out: 16×250) = **30256 < 32768** ✓

---

### RU-3 — Move A_H_STATE to A_BOT_OUT region (free after inception)

**Lifecycle of A_BOT_OUT (4000):**  
Used only during inception phase (bottleneck output). After S_DONE for inception, it is dead for the rest of the block.

**A_H_STATE** is only needed during M6A/M6B (SSM scan), which runs AFTER inception is complete.

**Fix:** Set `A_H_STATE = A_BOT_OUT` (= 4000).  
H_STATE max size = 256 words (block 4) → region [4000, 4256). A_BOT_OUT max size = 2000 words → the region [4000, 6000) easily contains H_STATE.

**Benefit:** Frees up the [28000, 28256) region, allowing A_MAMBA_OUT to use 28000 cleanly.  
New layout:
```
A_H_STATE   = 4000    # reuses dead A_BOT_OUT region post-inception
A_MAMBA_OUT = 28000   # free after RU-2 fix; simplifies parameterization
```

**Implementation notes:**  
- Replace `localparam A_H_STATE = 15'd28000;` with `localparam A_H_STATE = A_BOT_OUT;`
- Ensure no state reads A_BOT_OUT after inception AND before M6A (scan through FSM transitions to verify the gap)
- For block 4, A_BOT_OUT stores 2000 words (d_out=128, T=250) and H_STATE needs 256 — so they must not overlap: since A_BOT_OUT ends use at inception_done and H_STATE starts at M6, they are time-disjoint, not space-disjoint. The FIX requires the addresses to be different enough that A_BOT_OUT write region [4000, 6000) and A_H_STATE write region [4000, 4256) coexist — but they CAN'T coexist in the same bank at the same time unless we ensure the inception phase completes before any H_STATE write begins.  
  **Actually, they do coexist in space**, and since they're temporally disjoint (different FSM phases), this is safe — exactly the same reasoning as the current A_MAMBA_OUT overlapping A_H_STATE.

---

### RU-4 — Formally document A_X_INNER lifetime (no action, documentation only)

A_X_INNER (12000) holds SiLU(x_conv) (u) from M3 through M6 (end of SSM). After M7 reads from A_X_INNER (via the M3CP → B_U_SAFE → M6 chain, or directly after RU-1 fix), A_X_INNER is dead for the rest of the block. It can be reused for future passes.

---

## 4. Priority order

| Priority | Item | Impact | Complexity |
|----------|------|--------|------------|
| 1 | RU-1: Remove M3CP stage | ~140K cycles / inference | Low — address change + state removal |
| 2 | CP-2: Pipeline norm_sq16_fn | Fmax: deepest path | Medium — 4 substates, no logic change |
| 3 | CP-1: Pipeline x_norm_fn | Fmax: cascaded multiplies | Medium — new substep in M1A/M1B |
| 4 | CP-4: Register address strides | Fmax: removes 4 combinational mults | Low — always @(posedge) replace |
| 5 | RU-2+3: Fix A_MAMBA_OUT/A_H_STATE | Bug prevention + cleaner layout | Low — localparams only |
| 6 | CP-3: Pipeline S_FIN_WRITE bn_relu | Fmax: 16-lane multiply chain | Medium — need mul_reg(16×32b) |

---

## 5. Notes on simulation after changes

- After any FSM cycle count change (M3CP removal, norm_sq pipelining): re-run xsim TB with existing golden files — golden files do NOT need regeneration since data values are unchanged.
- After any address change (RU-2, RU-3): regenerate goldens (`python extract_itm_full.py --out ./golden_all --all_blocks`) and re-run verify_byte_exact.py to confirm RTL address match — addresses are not visible to the Python extractor, so only RTL xsim validates address correctness.
- CP-1/CP-2/CP-3/CP-4 are functional no-ops (same arithmetic, just pipelined) — existing goldens remain valid.
