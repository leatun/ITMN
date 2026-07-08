# Mamba_Top Per-Timestep Refactor

## Mục tiêu

Refactor `Mamba_Top` từ batch mode (chạy toàn bộ T timesteps cho từng stage
rồi sang stage kế) sang per-timestep pipeline (RMSNorm → M1A → M1B → M2 →
M3 → M4 → M5 → M6 → M7 → M8 chained cho mỗi t). Reuse RAM regions qua các
timestep bằng lifetime-based aliasing.

## Kết quả

`tb_Mamba_Top_FULL.v` — 1000 timesteps × 64 output channels = **64000 compares,
0 errors, byte-exact vs `Mam_OutProj_FP.txt`**.

## Kiến trúc

### FSM sequence per timestep

```
S_H_INIT (once) → for t in 0..T-1:
    RMSNorm → M1A → M1B → M2 → M3 → M4 → M5 → M6 → M7 → M8
```

`run_stage` port giữ lại để backward-compat nhưng ignored — top luôn chạy full
pipeline. `T_MAX` set số timestep.

### Memory layout compact (Slot-based aliasing)

Chỉ 1 URAM `ram_main` (4128 × 256-bit), không bank_sel. Layout `PT_*` trong
`_parameter.v`:

| Slot | Addr range | Contents (disjoint lifetime) |
|------|-----------|------------------------------|
| 0 | 0..7 | X_NORM → X_PROJ → Y_GATED |
| 1 | 16..31 | X_CONV → U → Y_SSM |
| 2 | 32..47 | DELTA (không aliased) |
| 3 | 48..63 | Z_GATE (giữ từ M1B đến M7) |
| CIRC | 64..127 | X_INNER_CIRC (4-tap circular, addr = (t%4)*CH_M + c_grp) |
| BULK | 128..4127 | INPUT ↔ MAMBA_OUT aliased (INPUT[t] chết trước khi MAMBA_OUT[t] ghi) |

Tổng 4128 words × 32 bytes = 132 KB, fit 1 URAM cascade × 5-deep.

### X_PROJ layout — B/C word-boundary handling

M4 ghi `X_PROJ[0..n_pad-1]` = `[dt(dt_rank), B(d_state), C(d_state), pad]`
packed 16-per-word. Với `dt_rank=4`, layout:

```
word 0 = [dt0..dt3 | B0..B11]
word 1 = [B12..B15 | C0..C11]
word 2 = [C12..C15 | 0..0]
```

M6 nạp cả 3 word raw rồi shuffle wire tái tạo B/C aligned:

```verilog
m6_dt_shift = DT_RANK << 4                    // = 64 for DT_RANK=4
m6_dt_comp  = 256 - m6_dt_shift               // = 192
m6_B_word   = (m6_w1_reg << m6_dt_comp) | (m6_w0_reg >> m6_dt_shift)
m6_C_word   = (m6_w2_reg << m6_dt_comp) | (m6_w1_reg >> m6_dt_shift)
```

Cost: +3 cycles/timestep cho load w0.

## Files

| File | Vai trò |
|------|---------|
| `sources_v3/common/_parameter.v` | PT_* slot bases + PE modes + block-0 dims |
| `sources_v3/common/Memory_System.v` | Single URAM main + 2 BRAM weight mirrors |
| `sources_v3/top/Mamba_Top.v` | Per-t FSM (10 stages chained) |
| `sources_v3/tb/tb_Mamba_Top_FULL.v` | End-to-end, so với `Mam_OutProj_FP.txt` |
| `sources_v3/tb/tb_Mamba_Top_DBG.v` | 1-timestep dump từng stage cho debug |
| `sources_v3/constrs/mamba_top.xdc` | 200 MHz OOC timing constraint |

## Test flow

```
1. tb_Mamba_Top_DBG (T=1) — dump X_INNER/Z_GATE/DELTA/Y_SSM/Y_GATED/MAMBA_OUT
   vs goldens tương ứng. Isolate stage-level bug nhanh.
2. tb_Mamba_Top_FULL (T=1000) — full sweep 64000 compares.
```

**Input golden**: `P1_Output_Golden_FP.txt` (raw block input = RMSNorm
input). KHÔNG dùng `P1_Norm_Output_FP.txt` (đó là RMSNorm output — nếu load
nhầm sẽ khiến RMSNorm áp dụng 2 lần).

## Debug history

- **Bug 1**: M6 đọc `PT_X_PROJ+1` như B, `PT_X_PROJ+2` như C — nhưng raw
  layout có B/C straddle word boundaries. Fix: shuffle 3 word.
- **Bug 2**: TB load nhầm `P1_Norm_Output_FP` (post-RMSNorm) làm INPUT.
  Sửa thành `P1_Output_Golden_FP` (pre-RMSNorm).

## Synth expectation (200 MHz KV260)

Cần chạy OOC synth để confirm WNS ≥ 0. Baseline batch-mode đạt 100 MHz với
WNS=0.646ns; per-t design có thêm 3-word shuffle mux nhưng ít stage
transition hơn, expect competitive fmax.
