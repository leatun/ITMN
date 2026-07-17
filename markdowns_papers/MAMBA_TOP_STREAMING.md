# Mamba_Top streaming pipeline — design notes

Companion to `sources_v3/top/Mamba_Top.v` after the P0–P5 streaming refactor pass. All comments were stripped from the RTL; this document is the source of truth for design rationale, state semantics, and per-block cycle costs.

## 1. Pipeline shape

Per timestep `t`, the FSM chains stages in a fixed order:

```
S_H_INIT (once at boot) -> RN_SQ -> RN_AP -> M1A -> M1B -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8 -> (t++ -> RN_SQ)
```

`cur_stage` tags the current logical stage (0..9) so the H_RegFile control mux (`h_ctrl_active`) can gate H_RegFile ports to M6 only. All intermediates live in URAM ram_main at compact `PT_*` slot bases (from `_parameter.v`); parameters in BRAM ram_weight at `W_*` bases; scalars/biases in const RAM at `C_*` bases.

## 2. Memory latency (assumed throughout)

| Resource            | NBA -> data valid |
|---------------------|-------------------|
| URAM ram_main (`m_rd_addr` -> `m_rd_data`)   | 2 cycles |
| BRAM ram_weight (`w_rd_addr` -> `w_rd_data`) | 2 cycles |
| Const_Storage (`const_rd_addr_r` -> `const_rd_data`) | 2 cycles |
| H_RegFile (`m6_h_rd_addr_r` -> `h_rd_data`)  | 1 cycle (SDPR)  |
| PE cluster (`cl_in_*` NBA -> `cl_out_vec`)   | 2 cycles (input reg + MAC + output reg) |
| Exp_LUT, SiLU_LUT, Softplus_LUT              | combinational (used mid-cycle from a registered PE output) |
| `rsqrt` LUT                                  | 2 cycles         |

Address issued at cycle T (NBA) -> register updates at T+1 rising edge -> RAM reads at T+1 -> output data at T+2 (URAM/BRAM/const). PE output register updates at T+2 for inputs NBA'd at T. Design everything against these deltas.

## 3. State groupings and cycle costs

All costs are for the B0 config: `d_model=64`, `d_inner=256`, `d_state=16`, `DT_RANK=4`. Groups iterate `c_out_grp` from 0..N-1; per stage N = mac_grp_count (M1/M4/M8) or inner_grp_last+1 (M2/M3/M6/M7).

### 3.1 Shared MAC skeleton (M1A / M1B / M4 / M8) — `S_PREFETCH..S_MAC_LATCH`

The MAC states are shared across four stages. `cur_stage` selects data base (`mac_in_base`), weight base (`mac_w_base`), write address (`mac_wr_addr`), inner length (`mac_len`), and group count (`mac_grp_count`).

Per-group state sequence:

```
PREFETCH -> WAIT -> MAC_LOAD -> MAC (loops mac_len/2 times using MAC2) -> MAC_LATCH
```

- `PREFETCH`: issue initial `m_rd_addr`, `w_rd_addr`, `w_rd_addr2`.
- `WAIT`: advance addresses one pair (T+2 pair). Clear c_in_cnt.
- `MAC_LOAD`: capture pair 0 into PE input regs. Prime accumulator (`cl_clear_acc=1`). Advance address two more pairs ahead.
- `MAC`: streaming loop. Each cycle captures the next pair into `cl_in_*` while PE consumes the previous pair. Exit when `c_in_cnt == mac_last2 + 2 == mac_len` (one extra cycle after the last real pair to let PE register the final result).
- `MAC_LATCH` (fused with the old `S_NEXT`): fire `m_we`, write `cl_out_vec` at `mac_wr_addr`, then in the same cycle branch:
  - If last group -> chain to next stage (M1A->M1B->M2, M4->M5, M8->next t or DONE).
  - Else -> increment `c_out_grp`, reissue next group's initial addresses, jump to `S_WAIT` (skip `S_PREFETCH` because addresses are already reissued).

Per-group cost: `4 + mac_len/2` cycles. Optimization vs pre-refactor: `MAC_LATCH+NEXT` fusion saves 1 cycle per group. Save 52 cycles/timestep across 16+16+16+4 groups (M1A, M1B, M4, M8 respectively for B0).

Notes:
- `cl_in_*_vec` are registered every cycle to break the URAM -> y_lane_sel -> mac_bypass -> DSP combinational chain that failed 100 MHz closure. The prefill happens in `MAC_LOAD`.
- Final write reads `cl_out_vec` (PE-registered) not `cl_out_next_vec` (combinational) so the URAM -> mult -> sat -> wr path stays under one clock period.

### 3.2 RMSNorm — `S_RN_SQ_* / S_RN_S_LATCH / S_RN_AP_*`

Two passes:

**SQ pass** (sum of squares -> mean -> rsqrt -> S):
```
RN_SQ_PREF -> RN_SQ_WAIT -> RN_SQ_MAC (streaming, d_model/16 iters) -> RN_SQ_FINAL -> RN_SQ_DONE -> RN_RSQ_WAIT -> RN_S_LATCH
```
- MAC accumulates `x[i]^2` with `cl_op_mode=MAC` and `cl_in_W1_vec=cl_in_H_ext=m_rd_data`.
- Sum reduced by `u_rw` (Reduce16Wide) into `sum_d_wide`. Mean is `sum_d_wide >>> rn_total_shift` where `rn_total_shift = log2_dm + 19` accounts for FRAC_BITS + integer shift for sqrt scaling. Clipped to 13 bits.
- Rsqrt ROM lookup (`u_const.rsqrt_data`) returns `S = 1 / sqrt(mean)` in the norm's fixed-point format. Captured to `S_reg` in `RN_S_LATCH`.

**AP pass** (apply norm, streamed across d_model/16 groups):
```
RN_AP_PREF -> RN_AP_WAIT -> RN_AP_MUL1 -> RN_AP_WAIT1 -> RN_AP_MUL2 -> RN_AP_WAIT2 -> RN_AP_WRITE (-> MUL1 for next group, or -> S_PREFETCH for M1A)
```
- Iter 0: full 7-cycle path (PREF, WAIT, MUL1, WAIT1, MUL2, WAIT2, WRITE).
- Iter j>=1: 5-cycle path — next iter's `x` and `g` reads are prefetched inside iter j-1's `MUL2` state (memory & const ports are idle then). `WRITE` folds the loop-back / stage-transition decision.
- MUL1: `x * g` (per-channel scale from norm weight).
- MUL2: `(x*g) * S_broadcast` (multiply by rsqrt scalar).

### 3.3 M2 depth-wise conv1d + bias — `S_M2_*`

4-tap circular conv. `X_INNER_CIRC` stores the last 4 timesteps' post-norm inputs indexed by `t_cnt[1:0]`.

Per group:
```
M2_PREF -> M2_WAIT -> M2_TAP (loops 4 taps, streaming) -> M2_FINAL -> M2_BIAS_PREF -> M2_BIAS_WAIT -> M2_BIAS_ADD -> M2_BIAS_LATCH -> M2_WRITE (fused NEXT)
```
- `M2_PREF/WAIT`: issue tap0/tap1 addresses.
- `M2_TAP`: at each cycle, capture w[tap] & x[tap] into PE inputs, run MAC (clear_acc on tap 0), advance addresses two ahead. `m2_pad` handles boundary (negative effective t).
- `M2_FINAL`: 1-cycle wait for PE to register final tap's MAC result into `cl_out_vec`.
- `M2_BIAS_PREF`: latch `cl_out_vec` -> `cl_in_W1_vec` and issue bias address on const RAM.
- `M2_BIAS_WAIT`: const RAM read latency.
- `M2_BIAS_ADD`: PE_ADD, `cl_in_H_ext <= const_rd_data`.
- `M2_BIAS_LATCH`: 1 cycle for PE output register.
- `M2_WRITE` (fused NEXT): write `cl_out_vec` (= acc + bias) at `m2_wr_addr`, then in same cycle branch to M3 (last group) or S_M2_PREF (next group).

Per group cost: `2 + 4 + 5 + 1 = 12` cycles (PREF/WAIT, 4 taps, FINAL/BIAS_PREF/WAIT/ADD/LATCH, WRITE). Save 1/group vs pre-refactor.

**Why FINAL cannot fuse with BIAS_PREF**: PE has 2-cycle latency, so `cl_out_vec` at FINAL still holds the tap_(N-2) partial sum. BIAS_PREF at FINAL+1 sees the fresh full-sum value. Fusing loses byte-exactness. Same constraint applies in M5 (see below).

### 3.4 M3 SiLU streaming — `S_M3_STREAM`

Single self-loop, `inner + 2` cycles per timestep. Three staggered stages inside one FSM state:

- Stage 1 (`c_out_grp <= inner_grp_last`): issue `m_rd_addr = PT_X_CONV + c_out_grp` for iter c_out_grp.
- Stage 2 (implicit): 2-cycle URAM latency puts iter (c_out_grp - 2)'s data on `m_rd_data`. `silu_in_drv` is combinationally routed to `m_rd_data` (via the state == S_M3_STREAM guard) so `silu_out_w` reflects `silu(m_rd_data)`.
- Stage 3 (`c_out_grp >= 2`): write `silu_out_w` at `PT_U + m3_wr_ptr`. `m3_wr_ptr` chases c_out_grp by 2.

When `m3_wr_ptr == inner_grp_last`, all writes are queued; jump to `S_PREFETCH` for M4. `c_out_grp <= 0` (the exit branch overrides the same-cycle increment thanks to NBA last-assignment-wins).

### 3.5 M5 dt_proj + bias + softplus — `S_M5_*`

Only reads `dt_raw` word once per timestep (single-word PT_X_PROJ slot 0). Per group iterates `m5_w_cnt` from 0..DT_RANK-1 doing dt[.] * W_dt[c, .] MAC, then adds bias, then applies softplus at write time.

```
M5_DT_READ -> DT_WAIT -> DT_LATCH   (once/timestep)
per group:
M5_W_PREF -> W_WAIT -> M5_MAC (loops DT_RANK iters) -> FINAL -> BIAS_PREF -> BIAS_WAIT -> BIAS_ADD -> BIAS_LATCH -> WRITE (fused NEXT)
```

- `M5_MAC` streaming: at each cycle capture next w into `cl_in_W1_vec`, MAC into accumulator, advance addresses two ahead. Uses `m5_dt_broadcast` on H input (dt scalar tapped from `m5_dt_word_reg[m5_dt_lane*16 +: 16]`).
- `M5_WRITE` writes `sp_out_w` (softplus of `cl_out_vec` fed combinationally). Fuses group increment / next-stage transition.

Save 1/group vs pre-refactor.

### 3.6 M6 SSM scan — `S_M6_*`

The biggest block. Per timestep: load B/C shared parameters once, then per channel-group loop over `c_out_grp` in 0..d_inner/16 - 1, and inside each group loop over `m6_lane` in 0..15 (state dimensions).

#### 3.6.1 Once-per-timestep W0/B/C load

X_PROJ layout is raw-concat `[dt(DT_RANK words), B(d_state), C(d_state), pad]` packed 16-per-word. Since `DT_RANK * 16 = 64` bits is not word-aligned (word = 256 bits), B and C straddle word boundaries. Load 3 raw words (w0, w1, w2) into regs, then reconstruct aligned B/C combinationally:

```
m6_B_word = (m6_w1_reg << m6_dt_comp) | (m6_w0_reg >> m6_dt_shift)
m6_C_word = (m6_w2_reg << m6_dt_comp) | (m6_w1_reg >> m6_dt_shift)
```
where `m6_dt_shift = DT_RANK * 16` and `m6_dt_comp = 256 - m6_dt_shift`.

Streamed load sequence (5 cycles vs 9 pre-refactor):

```
LOAD_W0_PREF (issue w0 addr)
LOAD_W0_WAIT (issue B addr)
LOAD_W0_LATCH (capture w0; issue C addr)
LOAD_B_LATCH (capture w1 = B raw)
LOAD_C_LATCH (capture w2 = C raw; c_out_grp <= 0; goto DT_PREF)
```

Timing works because URAM latency is 2 cycles and we're issuing back-to-back single-address reads on the same port; each LATCH captures the value that was addressed two cycles earlier.

#### 3.6.2 Per-group DT / U / D setup + lane-0 A/H prefetch

Streamed (4 cycles vs 9 pre-refactor):

```
LOAD_DT_PREF: issue dt addr (URAM), D addr (const), A[c(0)] addr (BRAM weight), H[c(0)] addr (RF)
LOAD_DT_WAIT: issue u addr (URAM)
LOAD_DT_LATCH: capture dt, capture D
LOAD_U_LATCH: capture u; m6_lane <= 0; goto DAB_MUL2
```

Because dt/u/D are on three different ports (URAM, URAM, const), they can be pipelined together. A/H for lane 0 are also prefetched here so the lane-loop can skip its own PREF/WAIT for lane 0.

Timing: DT_PREF at T -> dt/D data valid at T+2 (DT_LATCH); u addr fired at T+1 -> u data valid at T+3 (U_LATCH); A/H valid at T+2 -> stable through DAB_MUL2 at T+4 and T1_MUL later.

#### 3.6.3 Per-lane compute (17 cycles/lane after M6 optimization)

State chain (all lanes, including lane 0):

```
DAB_MUL2 -> EXP_WAIT -> DAB_LATCH
  -> T1_MUL -> T1_WAIT -> T1_LATCH
  -> T2_MUL -> T2_WAIT -> T2_LATCH
  -> H_ADD -> H_WRITE
  -> Y_MAC -> Y_WAIT -> Y_LATCH
  -> DU_MUL -> DU_WAIT -> DU_LATCH (fused FINALIZE + LANE_NEXT)
```

Meaning of each block:

- **DAB_MUL2**: PE MAC2 op — fuses dA (discretized A) preparation with dB. Inputs: W1=W2=`m6_dt_broadcast`, H=`w_rd_data`(= A[c]), X=`m6_B_word`(= B). Two outputs from PE:
  - `cl_out_vec = sat(dt * A >> FRAC_BITS)` -> feeds Exp_LUT.
  - `cl_out_vec2 = sat(dt * B >> FRAC_BITS)` = dB.
- **EXP_WAIT**: PE input reg / output reg staging.
- **DAB_LATCH**: capture `m6_dA_reg <= exp_out_w` (= exp(dt*A) = dA) and `m6_dB_reg <= cl_out_vec2` (= dB). Also assert `m6_h_from_rf_r <= 1` so the PE H-mux selects `h_rd_data` from the H_RegFile in T1_MUL.

  The Exp_LUT is fed the *registered* `cl_out_vec` (via `exp_in_drv`), not `cl_out_next_vec`, to break the URAM -> mult -> LUT -> `m6_dA_reg` combinational chain.

- **T1_MUL / WAIT / LATCH**: `m6_t1_reg <= dA * h[c]`. PE H input comes from `h_rd_data` (registered inside H_RegFile). Clear `m6_h_from_rf_r <= 0` at LATCH so subsequent states use `cl_in_H_ext`.
- **T2_MUL / WAIT / LATCH**: `m6_t2_reg <= dB * u_scalar` (u broadcast).
- **H_ADD**: PE ADD -> `t1 + t2` on `cl_out_vec` two cycles later.
- **H_WRITE**: `m6_h_wr_en_r <= 1`, `m6_h_wr_addr_r <= c`, `m6_h_wr_from_pe_r <= 1` — commits the new `h_new[c]` into H_RegFile from the PE's registered output.
- **Y_MAC / WAIT / LATCH**: SSM output MAC: `m6_y_ch_reg <= sat(sum(C[.] * h_new[.]))` per state dimension, using `cl_out_vec` as W1 and `m6_C_word` as H. `cl_clear_acc = 1` to reset accumulator per lane.

  **Cross-lane prefetch is inserted here**: if `m6_lane != 15`, issue `w_rd_addr = A[c(lane+1)]` on BRAM and `m6_h_rd_addr_r = c(lane+1)` on RF. Both land well before lane j+1's DAB_MUL2 (5+ cycles later).

- **DU_MUL / WAIT / LATCH (fused)**: `D * u` scalar. Fused DU_LATCH does the final ssm-accumulator write, the lane counter update, and the loop-back:
  - `m6_ssm_grp_acc[lane*16 +: 16] <= m6_y_ssm_scalar` where the scalar is `sat(m6_y_ch_reg + cl_out_vec[15:0])` computed via the combinational wire `m6_du_now = cl_out_vec[15:0]`. This eliminates the pre-refactor `m6_du_reg` intermediate.
  - If lane 15 -> `state <= WRITE_GRP`. Else -> `m6_lane++`, `state <= DAB_MUL2` (skip A_PREF / A_WAIT / H_PREF / H_WAIT thanks to Y_MAC prefetch).

#### 3.6.4 Group tail — `S_M6_WRITE_GRP` (fused GRP_NEXT)

Single-state group write + transition:
- `m_we <= 1`, `m_wr_addr <= PT_Y_SSM + c_out_grp`, `m_wr_data <= m6_ssm_grp_acc` — writes the 16-lane packed word back to URAM.
- Branch: if last group -> transition to `S_M7_Z_PREF` (`cur_stage <= 4'd7`); else -> `c_out_grp++`, `m6_lane <= 0`, jump to `S_M6_LOAD_DT_PREF` for the next group's DT/U/D setup.

#### 3.6.5 Byte-exact preservation

Key invariants against the pre-refactor golden:

- `m6_ysum` now computes off `cl_out_vec[15:0]` directly (= what `m6_du_reg` would have captured one cycle later). At DU_LATCH, PE has registered D*u into `cl_out_vec` (2 cycles after DU_MUL). Same 16-bit sat/round applied. Byte-identical.
- Cross-lane prefetch does not overlap with any address that would be needed by the current lane's compute. BRAM weight port is unused between DAB_MUL2 (input latch) and next lane's DAB_MUL2; H_RegFile read port is unused between T1_MUL and next lane's T1_MUL.
- `m6_h_from_rf_r` sequencing: cleared to 0 in T1_LATCH so DAB_MUL2 sees `cl_in_H_ext = w_rd_data` (correct A); set to 1 in DAB_LATCH so T1_MUL sees `h_rd_data`.
- H_RegFile write of `h[c]` at H_WRITE commits before lane j+1's T1_MUL reads `h[c(j+1)]` (different addr, so read-write ordering does not matter, but write completes at Y_WAIT edge anyway).

### 3.7 M7 gating — `S_M7_*`

Y_gated[c] = Y_ssm[c] * SiLU(z_gate[c]).

Streamed within-iter:

```
Iter 0: Z_PREF -> Z_WAIT -> Z_LATCH -> Y_WAIT -> MUL -> LATCH -> WRITE          (7 cycles)
Iter j >= 1: Z_LATCH -> Y_WAIT -> MUL -> LATCH -> WRITE                          (5 cycles)
```

- **Z_LATCH** fuses Y_PREF: `silu_z_reg <= silu_out_w` (silu of the z data on `m_rd_data`), and simultaneously issues `m_rd_addr <= m7_y_addr` for the y read.
- **Y_WAIT** is the 2nd cycle of the y read.
- **MUL**: PE MUL `cl_out_vec <= m_rd_data (y) * silu_z_reg`. Also prefetches iter j+1's z read on `m_rd_addr <= m7_z_addr_p1` — memory port is idle in MUL / LATCH / WRITE, and z data lands on `m_rd_data` exactly at Z_LATCH of iter j+1 (3 cycles later).
- **LATCH**: 1 cycle for PE output register.
- **WRITE**: fires memory write of `cl_out_vec` at `m7_wr_addr`. Loop-back branch: last iter -> `S_PREFETCH` (M8), otherwise jump directly to `S_M7_Z_LATCH` (skip PREF/WAIT because z is already prefetched).

### 3.8 M8 output projection

Same as M1A/M1B/M4 via the shared MAC skeleton with `cur_stage = 4'd8`. Writes `MAMBA_OUT[t * CH_OUT + c]`.

## 4. FSM state encoding notes

The `state` field is 7 bits. Encoding is sparse (some slots are historical / unused) but not renumbered to avoid touching every literal. Unused localparams left in place for the same reason — they synthesize away.

Dead but retained localparam names:
- `S_NEXT` (7'd5) — folded into `S_MAC_LATCH`.
- `S_M2_NEXT` (7'd30) — folded into `S_M2_WRITE`.
- `S_M5_NEXT` (7'd47) — folded into `S_M5_WRITE`.
- `S_M6_A_PREF / A_WAIT / H_PREF / H_WAIT` — replaced by prefetch inside setup / Y_MAC.
- `S_M6_LOAD_B_PREF / B_WAIT / C_PREF / C_WAIT / U_PREF / U_WAIT / D_PREF / D_WAIT / D_LATCH` — replaced by streamed W0/B/C and DT/U/D chains.
- `S_M6_FINALIZE / LANE_NEXT / GRP_NEXT` — folded into DU_LATCH / WRITE_GRP.
- `S_M7_Y_PREF` (was 7'd51) — fused into `S_M7_Z_LATCH`.

## 5. Cycle budget summary (B0)

Per-timestep cycle counts after this refactor pass (all groups):

| Stage      | Groups | Per-group   | Setup / tail  | Total    |
|------------|--------|-------------|---------------|----------|
| RN_SQ + RSQ | -      | -           | 4 + d_model/16 + 5 | ~13     |
| RN_AP      | 4      | 5 (iter>=1) | 2 (iter 0)    | ~22      |
| M1A        | 16     | 4 + d_model/2 | -           | 576      |
| M1B        | 16     | 4 + d_model/2 | -           | 576      |
| M2         | 16     | 12          | -             | 192      |
| M3         | -      | -           | inner + 2 = 18 | 18      |
| M4         | 16     | 4 + d_inner/2 | -           | 2112     |
| M5         | 16     | 8 + DT_RANK | 3 (once)      | ~195     |
| M6         | 16     | 4 (setup) + 16*17 + 1 (write) | 5 (once)  | ~4437    |
| M7         | 16     | 5 (iter>=1) | 2 (iter 0)    | ~82      |
| M8         | 4      | 4 + d_inner/2 | -           | 528      |
| **Total per timestep** | | | | **~8770 cycles** |

Approximate savings across the full P0..P5 sequence (baseline was ~10450 cycles/timestep):
- P3 (M3 SiLU streaming): 46
- P4 (RN_AP streaming): 10
- P2 (M7 streaming): 62
- P0 (M6 lane compaction: cross-lane prefetch + FIN fusion): 1536
- Streaming pass (M1/M2/M5/M6/M8 in this doc): 164
- **Cumulative: ~1818 cycles/timestep saved (~17%)**

Not-yet-applied opportunities (higher risk):
- P5: fold bias into MAC accumulator seed instead of separate ADD/LATCH pair — requires PE `cl_clear_acc` variant that seeds from a bias register. Est. 32 cycles/timestep (M2 + M5).
- PE latency reduction: replace `cl_out_vec` sinks with `cl_out_next_vec` in states where timing allows. Very large potential but timing closure risk.
- M6 further compaction: eliminate T1_WAIT / T2_WAIT / DU_WAIT / Y_WAIT / EXP_WAIT by using `cl_out_next_vec` at LATCH states — needs staged verification since each WAIT is there to defeat a specific comb chain.

## 6. Timing invariants (must hold after any future edit)

1. **Never assign `cl_in_*_vec` and read `cl_out_vec` in the same cycle** unless separated by at least 2 states (PE registers input + output).
2. **`m6_h_from_rf_r = 0` during DAB_MUL2** and any state that assigns `cl_in_H_ext` and expects the PE to use the external value.
3. **`m_we <= 1'b1` is a one-shot** — the always-block default clears it to 0. Fusing WRITE + NEXT is safe because m_we fires on the transition edge regardless of destination state.
4. **`c_out_grp` incremented in the same cycle as a memory write** works because the wire-based address expressions (`mac_wr_addr`, `mac_in_base`, etc.) evaluate using the *current* register value; the increment fires at the next edge.
5. **Cross-lane prefetch addresses must be issued at least 2 cycles before the next lane's DAB_MUL2** (for BRAM/URAM) or 1 cycle before its T1_MUL (for H_RegFile). Y_MAC is the canonical prefetch point.
6. **BIAS_PREF (M2, M5) cannot fuse with FINAL** — FINAL is the 1-cycle PE-latency wait for the last TAP MAC to register. Attempting fusion latches a partial sum. Save the cycle only if the PE gains a `cl_out_next_vec` bypass and the surrounding timing analysis confirms slack.

## 7. Test workflow

- After any FSM edit, run the M-block bit-exact TB in `xsim` and diff outputs against `P8_Mamba_Out_Golden_FP`. `mismatch_cnt` must be 0.
- Regenerate golden via `extract_itm_full.py` only if the Python model changed. Do not modify RTL and Python in the same commit.
- Per-t TBs load `P1_Output_Golden_FP` (pre-RMSNorm) — inputs to the pipeline are pre-norm. Do not confuse with `P1_Norm_Output_FP` used by standalone M1AB tests.
