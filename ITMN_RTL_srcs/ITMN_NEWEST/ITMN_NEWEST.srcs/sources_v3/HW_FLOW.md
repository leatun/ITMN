# ITMN Hardware Flow (Netron-style)

Version: Phase 6 backport + dense bank_B optimization (2026-08-02).
Design: `sources_v3/` — `Mamba_Top.v` + `Memory_System.v` + `M_Cluster.v` + `H_RegFile.v` + `LUT_Bank.v`.

Format: mỗi "layer" = **{Input vectors → Operation → Output vectors → HW mapping}**.
Bám sát RTL states từ `Mamba_Top.v`. Vector width dùng Q4.11 fixed-point (`DATA_W=16`).

---

## 0. ITMN Top-level (per record, PTB-XL 10s ECG)

Software topology (`ecg_models/ITMN.py`). Hardware chỉ xử lý phần **Mamba branch**; Inception + wrappers chạy trên khối riêng.

```
Input (B, 12, 1000)  ← 12-lead ECG, 100 Hz × 10s
   |
   ├─ Conv1d (12→d_model=64, k=1)  +  BatchNorm1d
   |     shape: (B, 64, 1000)
   |
   ├─ ITMBlock[0]  in=64  out=64   T=1000  → "B0"
   ├─ ITMBlock[1]  in=64  out=64   T=1000  → "B1"
   ├─ MaxPool1d(k=2, s=2)                  → T=500
   ├─ ITMBlock[2]  in=64  out=64   T= 500  → "B2"
   ├─ ITMBlock[3]  in=64  out=64   T= 500  → "B3"
   ├─ MaxPool1d(k=2, s=2)                  → T=250
   ├─ ITMBlock[4]  in=64  out=128  T= 250  → "B4"
   |
   ├─ Global Average Pool  → (B, 128)
   └─ Linear(128 → n_classes)  → logits
```

**Mỗi ITMBlock** = branch parallel:
```
        ┌── Conv1d 1x1 → BN → BaseInception (4-branch: MaxPool, k=9/19/39 convs)   → y_inc
   x ───┤
        └── Conv1d 1x1 → BN → transpose → RMSNorm → Mamba SSM → transpose → ReLU    → y_mam
                                                                            └─────────► sum → ReLU → out
```

**Hardware scope**: `Mamba_Top` chỉ compute nhánh **Mamba SSM** (RMSNorm → Mamba mixer). Inception + Conv1x1 + BN chạy pipeline khác. Golden reference: `itmn_pipeline.py` on WSL `/home/letun/ITMN_Latest/`.

---

## 1. Block dims (mapped to hardware runtime ports)

| Blk | d_model | d_inner | d_state | dt_rank | n_pad | T    | Groups (d_inner/16) |
|-----|---------|---------|---------|---------|-------|------|---------------------|
| B0  | 64      | 128     | 16      | 4       | 48    | 1000 | 8                   |
| B1  | 64      | 128     | 16      | 4       | 48    | 1000 | 8                   |
| B2  | 64      | 128     | 16      | 4       | 48    | 500  | 8                   |
| B3  | 64      | 128     | 16      | 4       | 48    | 500  | 8                   |
| B4  | 128     | 256     | 16      | 8       | 48    | 250  | 16                  |

**n_pad = dt_rank + 2*d_state**, ceil to 16-multiple (48). Runtime ports: `CH_OUT` (d_model/16), `CH_M` (d_inner/16), `XP_OUT_GRP` (n_pad/16 = 3), `DT_RANK`.

---

## 2. Mamba_Top per-timestep FSM (10 stages)

Mỗi timestep `t ∈ [0, T-1]` xử lý sequential 10 stage. Ping-pong dual-cluster (c1 + c2) áp dụng ở M1A/M1B/M8. Parallel M6 (c1 evens + c2 odds).

### Stage RN — RMSNorm (S_RN1 → S_RN13)

| Field | Value |
|---|---|
| **Input** | `x[d_model]` @ Q4.11 từ `PT_INPUT + t*CH_OUT` (ram_main) |
| **Weight** | `gamma[d_model]` @ Q4.11 từ ram_const `C_W_NORM_BASE` |
| **Op** | `sq_sum = Σ x[i]²` → `mean = sq_sum / d_model` → `rsqrt = 1/√(mean + eps)` → `y[i] = x[i] * rsqrt * gamma[i]` |
| **Output** | `x_norm[d_model]` @ Q4.11 → `PT_X_NORM` (ram_main slot 0) |
| **HW** | Reduce16 tree (sq_sum), MUL2 pipe (rsqrt scale), **RSqrt_ROM** (`LUT_Bank.u_rsqrt`, 1-cyc lat) |
| **Cluster** | u_mc single (RN không parallel — SQ reduce = scalar) |
| **Cycle** | ~294 cyc (RN loop d_model/16 iterations) |

### Stage M1A — InProj X (S_MAC1 → S_MAC5 → S_MAC5_B **[ping-pong]**)

| Field | Value |
|---|---|
| **Input** | `x_norm[d_model]` broadcast qua both clusters |
| **Weight (c1)** | `W_InProj_X_even[16 × d_model]` @ Q4.11 từ `ram_weight_A` bank at `W_INPROJ_X_BASE + g_even*d_model + k` |
| **Weight (c2)** | `W_InProj_X_odd[16 × d_model]` từ `ram_weight_B` (**DENSE**) at `W_INPROJ_X_BASE_B + (g/2)*d_model + k` |
| **Op** | `y[c] = Σ_k W[c,k] · x[k]` per output-channel c ∈ [0..d_inner) |
| **Output** | `x_inner[d_inner]` → `PT_X_INNER_CIRC + (t%4)*CH_M + g` (4-tap circular buffer for M2 depthwise) |
| **HW** | `M_Cluster` 16-lane × MAC2 (2 W×X ops/cyc/lane = 32 MAC/cyc/cluster = **64 MAC/cyc dual**) |
| **Cluster** | c1 even groups (0,2,4,...), c2 odd groups (1,3,5,...) — write serialize S_MAC5→S_MAC5_B via `cl2_out_snap` REG |
| **Cycle** | (d_inner/16)/2 × mac_len_iter + writeback = **~1840/5 stages** ≈ M1A ~370 cyc for B0 |

### Stage M1B — InProj Z (S_MAC1..MAC5_B, cur_stage=STG_M1B)

Same FSM as M1A, khác weight base + output slot.

| Field | Value |
|---|---|
| **Weight (c1/c2)** | `W_InProj_Z` từ `W_INPROJ_Z_BASE` / `W_INPROJ_Z_BASE_B` |
| **Output** | `z_gate[d_inner]` → `PT_Z_GATE` (giữ đến M7 gate) |
| **Ping-pong** | ✓ Same pattern as M1A |

### Stage M2 — Depthwise Conv 4-tap (S_M2_1 → S_M2_7)

| Field | Value |
|---|---|
| **Input** | 4 taps: `x_inner_circ[t-3..t] group g` từ `PT_X_INNER_CIRC + ((t-tap)%4)*CH_M + g` |
| **Weight** | `W_DW[d_inner × 4]` từ `W_DW_BASE + g*4 + tap` (SMALLS section bank_A) |
| **Bias** | `b_dw[d_inner]` từ ram_const `C_B_DW_BASE + g` |
| **Op** | `y[c] = Σ_{tap=0..3} W[c,tap] · x[t-tap, c] + b_dw[c]` — per-channel independent |
| **Output** | `x_conv[d_inner]` → `PT_X_CONV` (=`PT_U`, aliased) |
| **HW** | M_Cluster 16-lane MUL2 + bias fold (PREF fuse), no MAC2 (accumulation là scalar-per-lane) |
| **Cluster** | u_mc single (M2 sequential, không PP — chỉ 1 group/cyc write) |
| **Cycle** | ~7-9 cyc/group × groups ≈ 60-140 cyc |

### Stage M3 — SiLU (stream, no explicit FSM loop)

| Field | Value |
|---|---|
| **Input** | `x_conv[d_inner]` (từ PT_U) |
| **Op** | `u[c] = x[c] · sigmoid(x[c])` per element |
| **Output** | `u[d_inner]` in-place tại `PT_U` |
| **HW** | `LUT_Bank.u_silu` (Silu_LUT ROM, 2-cyc lat) — stream 16 lane/cyc |
| **Cycle** | ~groups + 2 cyc pipe fill |

### Stage M4 — X Projection (S_MAC1..MAC5, single-cluster)

| Field | Value |
|---|---|
| **Input** | `u[d_inner]` broadcast |
| **Weight** | `W_XProj[n_pad × d_inner]` từ `W_XPROJ_BASE + g_out*d_inner + k` (SMALLS bank_A) |
| **Op** | `y[c] = Σ_k W[c,k] · u[k]` for c ∈ [0..n_pad=48). n_pad = dt_rank + 2*d_state (padded to 48 for lane align) |
| **Output** | `xproj[n_pad]` → `PT_X_PROJ` (aliased với PT_X_NORM). Bao gồm: `dt_raw[0..dt_rank)`, `B[dt_rank..dt_rank+d_state)`, `C[dt_rank+d_state..n_pad)` |
| **HW** | u_mc single (n_pad/16 = 3 groups, odd → không hợp PP) |
| **Cluster** | u_mc only (c2 idle) |
| **Cycle** | 3 × (d_inner/16 * mac_iter) ≈ 100-300 cyc |

### Stage M5 — DtProj + Softplus (S_M5_*)

| Field | Value |
|---|---|
| **Input** | `dt_raw[dt_rank]` từ xproj (slice PT_X_PROJ[0..dt_rank)) |
| **Weight** | `W_DtProj[d_inner × dt_rank]` từ `W_DTPROJ_BASE + g*dt_rank + k` (SMALLS bank_A) |
| **Bias** | `b_dt[d_inner]` từ ram_const `C_B_DT_BASE + g` |
| **Op 1 (matmul+bias)** | `dt_lin[c] = Σ_k W[c,k]·dt_raw[k] + b_dt[c]` per channel c |
| **Op 2 (softplus)** | `delta[c] = log(1 + exp(dt_lin[c]))` |
| **Output** | `delta[d_inner]` → `PT_DELTA` |
| **HW** | M_Cluster MUL2 + bias fold + `LUT_Bank.u_softplus` (Softplus_LUT ROM, 2-cyc lat) |
| **Cluster** | u_mc single (M5 = 1 group only) |
| **Cycle** | ~50-100 cyc |

### Stage M6 — SSM Scan **[parallel dual-cluster]** (S_M6_1 → S_M6_19_B)

Đây là stage phức tạp nhất — per-channel SSM update sequential over `l ∈ [0..d_state)`.

| Field | Value |
|---|---|
| **Input per (g, l)** | `delta[c]`, `u[c]`, `B[l]`, `C[l]`, `D[c]`, `A_log[c,l]`, `h_prev[c,l]` |
| **Op 1: dA** | `dA[c,l] = exp(delta[c] · A_log[c,l])` — bit-exact matches PyTorch discretize |
| **Op 2: dB** | `dB[c] = delta[c] · B[l]` (l-outer, c-inner) |
| **Op 3: SSM update** | `h_new[c,l] = dA[c,l] · h_prev[c,l] + dB[c] · u[c]` |
| **Op 4: y_ch** | `y[c,l] = C[l] · h_new[c,l]` (16-lane per group) |
| **Op 5: y_sum** | `y_ssm[c] = Σ_l y[c,l] + D[c] · u[c]` (16-lane reduce → 1 scalar per c) |
| **Op 6 (fused M7 gate)** | `y_gated[c] = y_ssm[c] · SiLU(z_gate[c])` |
| **Output** | `y_ssm[d_inner]` → `PT_Y_SSM`, `y_gated[d_inner]` → `PT_Y_GATED` |
| **Loading (LOAD template)** | S_M6_1..M6_3: 5-slot load (w0, B, C at ctr_load=0; delta+D+A+h at ctr_load=3) |
| **DAB_MUL2** | S_M6_4: parallel Exp + Mul |
| **SSM chain** | S_M6_5..M6_10: MUL, ADD, LATCH h_new to `H_RegFile` |
| **Y_MAC** | S_M6_11..M6_13: `y_ch = C · h_new`, prefetch next lane |
| **DU_MUL/YSUM** | S_M6_14..M6_18: reduce Σ y + D·u |
| **WRITE_GRP** | S_M6_19 (c1) + S_M6_19_B (c2 serial) |
| **HW** | u_mc + u_mc2 (**dual cluster full**). Dual `LUT_Bank` (u_lut for c1 Exp/SiLU, u_lut2 for c2). Dual `H_RegFile` (H_A + H_B in URAM). |
| **Cluster** | c1 evens, c2 odds — mỗi cluster own H state |
| **Cycle** | ~loops(d_inner/16 /2) × 16(d_state) × per-lane-cyc + write serialize ≈ **1000-4000 cyc**, dominant stage |

**Bank_B DENSE addressing cho c2**:
- `w_grp_base_c2 = mac_wB_base + (ctr_g>>1) * mac_len_ext`  (thay vì `mac_w_base + ctr_g_p1 * mac_len_ext`)
- c2 W_A read: `W_A_BASE_B + {ctr_g_p1[3:1], 4'd0}` — dense odd index

### Stage M7 — Gate (merged into M6 tail)

Từ Phase 6: M7 elementwise gate `y_gated[c] = y_ssm[c] · SiLU(z_gate[c])` **merged inline sau S_M6_19_B**. Không còn state riêng.

| Field | Value |
|---|---|
| **Input** | `y_ssm[d_inner]`, `z_gate[d_inner]` từ PT_Z_GATE |
| **Op** | Elementwise MUL với SiLU(z) từ LUT_Bank |
| **Output** | `y_gated[d_inner]` → `PT_Y_GATED` (aliased với PT_X_NORM) |
| **Cycle** | 0 (merged) |

### Stage M8 — OutProj **[ping-pong]** (S_MAC1..MAC5_B)

| Field | Value |
|---|---|
| **Input** | `y_gated[d_inner]` broadcast |
| **Weight (c1)** | `W_OutProj_even[16 × d_inner]` từ bank_A `W_OUTPROJ_BASE + g_even*d_inner + k` |
| **Weight (c2)** | `W_OutProj_odd[16 × d_inner]` từ bank_B **DENSE** `W_OUTPROJ_BASE_B + (g/2)*d_inner + k` |
| **Op** | `y[c] = Σ_k W[c,k] · y_gated[k]` for c ∈ [0..d_model) |
| **Output** | `mamba_out[d_model]` → `PT_MAMBA_OUT + t*CH_OUT + g` (per-timestep append) |
| **HW** | Same as M1A/M1B (ping-pong MAC2) |
| **Cluster** | c1 evens + c2 odds |
| **Cycle** | ~(d_model/16)/2 × d_inner/16 * mac_iter, then loop to next `t` |

---

## 3. Memory topology (Phase 6 optimized)

```
Memory_System
├── ram_main_c1   URAM 4128×256b (SDP) — c1 dedicated read (port_b), fanout write
├── ram_main_c2   URAM 4128×256b (SDP) — c2 dedicated read (port_b), mirror write from same bus
│                  Purpose: dual-port parallel core read (mac in/out, SSM state buffers)
│
├── ram_weight_A  BRAM 8192×256b (TDP) — SMALLS (W_DW, W_XProj, W_DtProj, W_A) + PP even weights
│                  Port_b = c1 W1 read; Port_a = c1 W2 read | DMA write
│                  Layout: [0..1216) SMALLS, [1216..3264) X, [3264..5312) Z, [5312..7360) OUT, [7360..8192) spare
│
├── ram_weight_B  BRAM 4096×256b (TDP, **DENSE**) — odd-only PP weights + odd W_A
│                  Port_b = c2 W1 read; Port_a = c2 W2 read | DMA write
│                  Layout: [0..128) W_A_odd, [128..1152) X_odd, [1152..2176) Z_odd, [2176..3200) OUT_odd
│
├── ram_const     LUTRAM 128×256b (distributed) — gamma, b_dw, b_dt, D_param
│                  Port_b = c1 read; Port_a = c2 read | DMA write
│
├── H_RegFile_A   URAM 256×256b — c1 SSM h_state (even groups)
└── H_RegFile_B   URAM 256×256b — c2 SSM h_state (odd groups)
```

**DMA target encoding (3-bit)**:
- 3'd0 = ram_main (fanout to c1+c2 mirrors)
- 3'd2 = ram_weight_A
- 3'd3 = ram_const
- 3'd4 = ram_weight_B

---

## 4. Compute topology (per cluster)

```
M_Cluster (u_mc / u_mc2)
├── 16-lane Mamba_PE array (SIMD)
│    Each lane = {Mult, Add40b, Sat16, Reg}
│    Op modes: MAC, MAC2 (2 W×X/cyc), MUL, MUL2, SSM (specialized)
│
├── H_RegFile (HAS_H=1) — 256×256b URAM per cluster
│    Stores h_state[c, s=0..15] per channel c
│
└── Reduce16Wide u_rw — 40b × 16 → 40b sum tree (for y_ssm sum)

Top-level shared:
├── LUT_Bank u_lut / u_lut2
│    ├── Silu_LUT   (SiLU ROM)
│    ├── Exp_LUT    (Exp ROM)
│    ├── Softplus_LUT
│    └── RSqrt_ROM  (single, RN only — non parallel)
│
└── DMA controller (streaming reload_req level signal)
```

---

## 5. Cycle budget per timestep (B0 example, d_inner=128)

| Stage | State range | Cyc/t (measured) | HW cluster |
|-------|-------------|------------------|------------|
| RN | S_RN1..RN13 | ~50 | u_mc single |
| M1A | S_MAC1..MAC5_B (pp) | ~250 | u_mc + u_mc2 ping-pong |
| M1B | S_MAC1..MAC5_B (pp) | ~250 | u_mc + u_mc2 ping-pong |
| M2 | S_M2_1..M2_7 | ~80 | u_mc single |
| M3 | stream | ~15 | LUT_Bank stream |
| M4 | S_MAC1..MAC5 | ~200 | u_mc single (3 grp odd) |
| M5 | S_M5_* | ~60 | u_mc single (1 grp) |
| M6 | S_M6_1..M6_19_B | **~700** | u_mc + u_mc2 parallel |
| M7 | merged in M6 | 0 | — |
| M8 | S_MAC1..MAC5_B (pp) | ~230 | u_mc + u_mc2 ping-pong |
| **Total** | — | **~1840** | — |

**5-block total** (T=1000/500/500/1000/250, dims varying): **6.68M cyc @ 100 MHz = 66.77 ms/record**.

---

## 6. Design invariants (byte-exact critical)

1. **Per-channel independence** — M1A/M1B/M4/M6/M8 MAC theo output-channel c độc lập → PP split evens/odds không đổi kết quả.
2. **Snap register c2** — `cl2_out_snap` capture ở S_MAC5 posedge để tránh extra-iter corruption khi write serial vào S_MAC5_B.
3. **BRAM 2-cyc latency** — S_MAC3/M4/M6_2 addr issue → dout arrive 2 cyc sau; FSM luôn giữ 1 WAIT gap giữa addr NB-assign và data consumer.
4. **Level DMA req** — `reload_req` phải LEVEL không PULSE (pump có thể busy khi TB gọi lại).
5. **Dense bank_B** — c2 addressing `(ctr_g>>1)*mac_len_ext + mac_wB_base` với bases riêng cho bank_B (`W_INPROJ_X_BASE_B/Z_B/OUT_B/W_A_BASE_B`).

---

## 7. Golden reference (Python)

- `extract_itm_full.py` → sinh golden per-stage vào `golden_all/BX/*.txt` (Q4.11 hex).
- `itmn_pipeline.py verify` → byte-exact check 85/85 mẫu.
- `itmn_pipeline.py eval` → AUC/TPR trên PTB-XL SUPER task.
- Kernel per-channel independent → không cần patch cho dual-cluster PP.

Sim TBs (Windows Vivado xsim):
- `tb_Mamba_Top_DBG.v` — T=1, single-block, stage-by-stage compare
- `tb_Mamba_Top_FULL.v` — T=1000, single-block full byte-exact
- `tb_Mamba_Top_5BLOCK.v` — all 5 blocks sequential + DMA reload, cycle report

---

**Version**: Phase 6 optimized (dense bank_B, LUTRAM const, URAM H_RegFile) — 2026-08-02.
**Synth**: WNS +0.096 ns @ 100 MHz, LUT 23.2k, BRAM 85, URAM 24, DSP 96, Power 0.614W (KV260).
**Throughput**: 14.97 records/s (Mamba only), 1.95× speedup vs old baseline.
