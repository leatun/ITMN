# Mamba_Top Per-Timestep Cycle Profile (Block 0)

Đếm chính xác cycle-count từ FSM trong `sources_v3/top/Mamba_Top.v`. Mỗi
state là **1 cycle** (Vivado NBA semantics, state transitions on posedge).
BRAM có 1-cycle read latency, WAIT states sinh ra từ đó.

## Configuration (Block 0)

| Param | Value | Ghi chú |
|-------|-------|---------|
| `d_model` | 64 | `CH_OUT = 4` |
| `d_inner` | 128 | `CH_M = 8` (ch_m_act × 16) |
| `d_state` | 16 | Vector width của M_Cluster |
| `dt_rank` | 4 | |
| `n_pad` | 48 | `XP_OUT_GRP = 3` groups |
| `T` | 1000 | `T_MAX` |
| MAC width | MAC2 | 2 elements/cycle (dual-DSP fused) |

## Cycle count per state group

### RMSNorm (`cur_stage=9`)

| Phase | States | Cycles |
|-------|--------|--------|
| Sum-of-squares | SQ_PREF(1) + SQ_WAIT(1) + SQ_MAC(× d_model/16 = 4) + SQ_FINAL(1) + SQ_DONE(1) + RSQ_WAIT(1) + S_LATCH(1) | **10** |
| Apply γ · rsqrt · x (per grp × 4 grps) | AP_PREF+WAIT+MUL1+WAIT1+MUL2+WAIT2+WRITE+NEXT = 8 cycles × 4 grps | **32** |
| **Total RMSNorm** | | **42** |

### M1A — x_in_proj (`cur_stage=0`)

MAC pattern: PREFETCH(1) + WAIT(1) + MAC(d_model/2 = 32) + NEXT(1) = 35 cycles/grp
- Number of output groups = `ch_m_act` = 8

**M1A = 8 × 35 = 280 cycles**

### M1B — z_in_proj (`cur_stage=1`)

Giống M1A hoàn toàn (cùng d_model → d_inner MAC).

**M1B = 280 cycles**

### M2 — depthwise conv 4-tap + bias (`cur_stage=2`)

Per group: PREF(1) + WAIT(1) + TAP(4) + FINAL(1) + BIAS_PREF(1) + BIAS_WAIT(1) + BIAS_ADD(1) + BIAS_LATCH(1) + WRITE(1) + NEXT(1) = 13 cycles
- 8 inner groups (ch_m_act)

**M2 = 8 × 13 = 104 cycles**

### M3 — SiLU elementwise (`cur_stage=3`)

Per group: PREF(1) + WAIT(1) + WRITE(1) + NEXT(1) = 4 cycles × 8 grps

**M3 = 32 cycles**

### M4 — x_proj (`cur_stage=4`)

MAC pattern: PREFETCH(1) + WAIT(1) + MAC(d_inner/2 = 64) + NEXT(1) = 67 cycles/grp
- Number of output groups = XP_OUT_GRP = 3

**M4 = 3 × 67 = 201 cycles**

### M5 — dt_proj + bias + softplus (`cur_stage=5`)

Fetch dt word (once): DT_READ(1) + DT_WAIT(1) + DT_LATCH(1) = 3 cycles

Per group: W_PREF(1) + W_WAIT(1) + MAC(dt_rank = 4) + FINAL(1) + BIAS_PREF(1) + BIAS_WAIT(1) + BIAS_ADD(1) + BIAS_LATCH(1) + WRITE(1) + NEXT(1) = 13 cycles × 8 grps = 104

**M5 = 3 + 104 = 107 cycles**

### M6 — SSM scan (`cur_stage=6`) — BOTTLENECK

X_PROJ shuffle (1 lần/timestep): LOAD_W0 (3) + LOAD_B (3) + LOAD_C (3) = **9 cycles**

Per group (8 groups):

| Sub-phase | Cycles |
|-----------|--------|
| LOAD_DT (delta[c_grp]) | 3 |
| LOAD_U (u[c_grp]) | 3 |
| LOAD_D (D[c_grp]) | 3 |
| **Lane loop (16 lanes × 21 states)** | **336** |
| WRITE_GRP + GRP_NEXT | 2 |
| **Per-group total** | **347** |

Chi tiết 21 states/lane:
```
A_PREF, A_WAIT, DA_MUL, DA_LATCH, DB_MUL, DB_LATCH,
H_PREF, H_WAIT, T1_MUL, T1_LATCH, T2_MUL, T2_LATCH,
H_ADD, H_WRITE, Y_MAC, Y_WAIT, Y_LATCH,
DU_MUL, DU_LATCH, FINALIZE, LANE_NEXT
```

**M6 = 9 + 8 × 347 = 2785 cycles**

### M7 — y_ssm × SiLU(z_gate) (`cur_stage=7`)

Per group: Z_PREF(1) + Z_WAIT(1) + Z_LATCH(1) + Y_PREF(1) + Y_WAIT(1) + MUL(1) + LATCH(1) + WRITE(1) + NEXT(1) = 9 cycles × 8

**M7 = 72 cycles**

### M8 — out_proj (`cur_stage=8`)

MAC pattern: PREFETCH(1) + WAIT(1) + MAC(d_inner/2 = 64) + NEXT(1) = 67 cycles/grp
- Number of output groups = CH_OUT = 4

**M8 = 4 × 67 = 268 cycles**

## Bảng tổng hợp per-timestep

| Stage | Cycles | % of t | MAC ops (Q4.11) |
|-------|-------:|-------:|----------------:|
| RMSNorm | 42 | 1.01% | d_model = 64 |
| M1A (x-in-proj) | 280 | 6.71% | d_model × d_inner = 8192 |
| M1B (z-in-proj) | 280 | 6.71% | 8192 |
| M2 (dw conv) | 104 | 2.49% | 4 × d_inner = 512 |
| M3 (SiLU) | 32 | 0.77% | 0 (LUT) |
| M4 (x-proj) | 201 | 4.82% | d_inner × n_pad = 6144 |
| M5 (dt-proj + softplus) | 107 | 2.57% | d_inner × dt_rank = 512 |
| **M6 (SSM scan)** | **2785** | **66.77%** | d_inner × d_state × 4 = 8192 |
| M7 (gate) | 72 | 1.73% | d_inner = 128 (mul) |
| M8 (out-proj) | 268 | 6.42% | d_inner × d_model = 8192 |
| **Total** | **4171** | **100%** | ≈ 40k MAC/t |

## Latency block-0 full run

| Metric | Value |
|--------|-------|
| Cycles/timestep | 4171 |
| T (block 0) | 1000 |
| Total cycles | 4.171 M |
| @ 100 MHz | 41.71 ms |
| @ 200 MHz | 20.86 ms |
| Compute throughput | 40k MAC / 4171 cycles ≈ 9.6 MAC/cycle |
| Peak PE utilization | 9.6 / 32 (MAC2 × 16 lanes) ≈ 30% |

## Chẩn đoán bottleneck

**M6 chiếm 66.8% (không phải 90%)** — dự đoán của user hơi cao nhưng đúng
hướng: M6 là bottleneck duy nhất chi phối > half thời gian.

Nguyên nhân M6 chậm:

1. **Serial 16 lanes/group × 8 groups = 128 lần**: M_Cluster hiện tại
   parallel 16-way trên d_state, nhưng lane loop (channel-within-group)
   serial hoàn toàn.
2. **21 cycles/lane**: chuỗi 5 phép MUL/ADD (DA, DB, T1, T2, H_ADD, Y_MAC,
   DU) mỗi phép mất 2-3 cycles (compute + latch) vì không pipeline giữa
   các phép.
3. **exp LUT dependency**: DA_LATCH cần chờ exp_out_w từ Const_Storage,
   ép serial DA → DB.

## Các stage khác

- **MAC stages (M1A/M1B/M4/M8) tổng 1029 cycles = 24.7%**: dominant sau M6.
  Peak MAC-throughput trong S_MAC là **2 element/cycle** (MAC2). Có thể
  tăng lên 4/8/16 nếu double/quadruple/16× PE. Overhead prefetch/wait/next
  chiếm 3/(2+d_len/2) mỗi group — không đáng kể khi mac_len lớn.
- **Elementwise stages (M2/M3/M7 = 208 cycles = 5%)**: không xứng đáng
  tối ưu, latency chủ yếu là RAM read-wait, không phải compute.
- **RMSNorm 42 cycles = 1%**: đã tối ưu tốt.

## Hướng tăng tốc (ước lượng speedup)

### A. Chỉ pipeline 2-stage (pre-SSM ∥ SSM+post), 0 thêm PE

- Pre-SSM per t: RN + M1A + M1B + M2 + M3 + M4 + M5 = **1046 cycles**
- SSM+post per t: M6 + M7 + M8 = **3125 cycles**
- Steady-state = max(1046, 3125) = **3125 cycles/t**
- **Speedup = 4171 / 3125 = 1.33×**
- Cost: double-buffer slots {X_INNER, X_CONV, U, DELTA, X_PROJ, Y_SSM,
  Z_GATE, Y_GATED, MAMBA_OUT} → +9 slots × 16 words = 144 words BRAM.
  FSM +1 bit `t_parity`. Không thêm DSP.
- Kết luận: **rẻ nhưng chỉ 33% speedup vì M6 vẫn nghẽn**.

### B. Parallel M6 across channel-lanes (K clusters)

Split 16 lanes trong 1 group thành K PE clusters song song. Mỗi cluster
vẫn giữ 16-way vector trên d_state (16 DSPs).

| K | M6 cycles | Total/t (no pipeline) | Speedup | Extra DSP |
|---|----------:|----------------------:|--------:|----------:|
| 1 | 2785 | 4171 | 1.0× | 0 |
| 2 | ~1440 | 2826 | 1.48× | +~20 |
| 4 | ~750 | 2136 | 1.95× | +~60 |
| 8 | ~410 | 1796 | 2.32× | +~140 |

### C. **Recommended: B + A combined**

Pipeline 2-stage + K=4 M6 parallel:
- Pre-SSM = 1046
- SSM+post ≈ 750 + 72 + 268 = 1090
- Steady state ≈ max(1046, 1090) = **1090 cycles/t**
- **Speedup = 4171 / 1090 = 3.83×**
- Total DSP ~ +60, BRAM double-buffer ~144 words

Sau đó pre-SSM (M1A/M1B/M4 mỗi cái ~280 cycles) trở thành bottleneck kế.
Muốn đẩy tiếp phải increase MAC width (MAC4/MAC8) trên các stage này —
thêm ~16-32 DSP nữa.

## Files

- Code: `sources_v3/top/Mamba_Top.v` (FSM, ~1100 dòng)
- Verified: `sources_v3/tb/tb_Mamba_Top_FULL.v` (64000 compares, 0 errors)
