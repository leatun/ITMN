# Mamba PE Redesign — Bottom-up Architecture

*Created: 2026-06-23 — Phase 2a precursor*

Tài liệu này phân tích workload Mamba ở mức **phép tính nguyên tử**, đề xuất 3-4 PE microarchitecture, chọn 1 hướng đi, và bỏ ngỏ điểm kết với Inception path (phục vụ A+B dual-cluster ở `plan.md`).

---

## 0. Vì sao bắt đầu từ PE?

Datapath là **đáy của ngăn xếp thiết kế**. Mọi quyết định ở FSM, memory, controller đều phải fit interface của PE. Sai PE → toàn hệ thống follow theo. Đặc biệt với Mamba SSM, PE generic hiện tại (`Unified_PE` 1-DSP, scalar×vector hoặc vec×vec) làm M6 scan tốn cycle bất hợp lý — đây chính là chỗ phải mổ.

Cách tiếp cận:

1. Đo workload Mamba xem **phép tính nào dominates** và **pattern data reuse** ra sao.
2. Đề xuất 3-4 PE option có trade-off rõ.
3. Chốt 1 PE → spec interface chính xác (input/output/mode/timing).
4. Phác cluster (8-lane M-cluster) wrap quanh PE.
5. Đề cập chỗ I-cluster (Inception) chia sẻ / khác biệt.

---

## 1. Mamba workload — đếm phép tính & xác định bottleneck

### 1.1 Pipeline Mamba per timestep (đã có từ `extract_itm_full.py`)

Input: `x_norm` shape `(d_in,)` sau RMSNorm. Block 4: `d_in=64, d_inner=256, d_state=16, dt_rank=8`. Block 0-3: `d_in=64, d_inner=128, d_state=16, dt_rank=4`.

| Stage | Phép tính | Shape mỗi t | Ops per t (B4) | Mode |
|-------|----------|-------------|---------------|------|
| M1A | x_inner = W_x · x_norm | (d_inner) = MAC reduction qua d_in | 256 × 64 = **16384** | MAC |
| M1B | z_gate = W_z · x_norm | (d_inner) | 16384 | MAC |
| M2 | x_conv = SiLU(DWConv1D(x_inner, k=4)) | (d_inner), depthwise | 256 × 4 = 1024 | MUL+ADD+SiLU |
| M3 | (Δ_raw, B, C) = W_proj · x_conv | (dt_rank + 2·d_state) = (8+32) | 256 × 40 = 10240 | MAC |
| M4 | Δ = softplus(W_dt·Δ_raw + dt_bias) | (d_inner) | 256 × 8 = 2048 | MAC + softplus |
| M5 | dA = exp(Δ ⊗ A), dB = Δ ⊗ B | (d_inner × d_state) | 256 × 16 × 2 = 8192 | MUL + exp |
| **M6** | **SSM scan**: h_t = dA·h_{t-1} + dB·x_conv; y_t = sum_s h_t[s,:]·C_t[s] | **(d_inner × d_state)** | **256 × 16 × 4 = 16384** | **dual-mul + acc** |
| M7 | y_gate = y · SiLU(z_gate) | (d_inner) | 256 + 256 SiLU | MUL + SiLU |
| M8 | mamba_out = W_out · y_gate | (d_out) | 128 × 256 = 32768 | MAC |

**Tổng ops per t (B4)**: ~115K. M1A+M1B+M8 = 65K (linear). M6 = 16K (recurrence). Còn lại = 34K (projections + element-wise).

**Cycle thực tế trên RTL hiện tại** (per t, B4, estimated từ FSM state count):

| Stage | Cycles per t | Bottleneck pattern |
|-------|--------------|-------------------|
| RMSNorm | ~80 | sum-square + rsqrt + multiply |
| M1A + M1B | ~120 | 2× MAC reduction qua 64 |
| M2 | ~30 | 4-tap depthwise, 16 channel group |
| M3 + M4 + M5 | ~200 | 3 projection nhỏ + 2 LUT |
| **M6 SSM scan** | **~700** | **per (t, s, c_group)**: read dA → mul → cap → read dB → mul → add → write h → read C → mul → accumulate. **Ăn ~50% cycle Mamba.** |
| M7 + M8 | ~150 | MUL + MAC out_proj |

→ **M6 là target tối ưu chính**. M1A/M1B/M8 đứng thứ 2 (cùng pattern MAC reduction, có thể reuse PE giống nhau).

### 1.2 Data reuse pattern của M6

```
for t in 0..T-1:                  # outer loop, serial (recurrence)
  for c_grp in 0..(d_inner/8)-1:  # mid loop, parallel-able along channels
    for s in 0..d_state-1:        # inner loop, parallel-able along state
      h_new[s, c_grp] = dA[t, c_grp, s] * h_old[s, c_grp]      # mul-1
                     + dB[t, c_grp, s] * x_conv[t, c_grp]      # mul-2
      y_partial[c_grp] += h_new[s, c_grp] * C[t, s]            # mul-3
    # after s loop: y[t, c_grp] = sat(y_partial[c_grp])
```

Observations:
- **h[s, c]** là state vector, read+write mỗi (t, s, c) → highest BW.
- **x_conv[t, c]** broadcast qua tất cả s (state). Cấp 1 lần per (t, c_grp).
- **C[t, s]** broadcast qua tất cả c. Cấp 1 lần per (t, s).
- **dA[t, c, s], dB[t, c, s]** unique per (t, c, s) — không reuse.
- **y_partial[c]** accumulate qua s → reduction tree theo s khi end loop s.

→ Pattern này KHÔNG fit "scalar broadcast + vector × vector" generic PE. Cần PE với **2 multiplier song song chia sẻ acc** + **h_state local storage**.

---

## 2. Đề xuất PE — 4 options

Mỗi option có sketch datapath + DSP count + cycle estimate cho M6 B4 (per timestep, T=250).

### Option PE-A — "Unified++" minimal: 1 DSP, add SSM-half mode

**Idea**: Giữ `Unified_PE` 1-DSP. Thêm 1 mode `SSM_HALF`: mỗi cycle compute 1 mul + 1 add với external operand A.

```
                          ┌────────────┐
in_A ──┐                  │ acc_raw    │
        ├─► DSP ──► +sat ─┤ 40-bit reg │
in_B ──┘   │   ▲          └────────────┘
           │   │
        clear_acc / accumulate sel
```

PE giữ nguyên 1 DSP + 40-bit acc. M-cluster phải **chia M6 thành 2 cycles per (s, c)**: cycle 1 compute `dA*h`, cycle 2 compute `dB*x + acc` → 1 result `h_new`. Cycle 3 compute `h_new*C` accumulate vào y.

**Cycle per t M6 B4**: 16 state × 32 c_grp × 3 = **1536** (worse than current — vì current đang share PE_Array 16-lane parallel theo c_grp).

→ **Loại**. Không cải thiện gì.

### Option PE-B — "DA-PE": 2 DSP fused SSM mul-add per cycle

**Idea**: Dedicated SSM PE với 2 DSP song song + adder + acc. 1 cycle finish `h_new = dA·h + dB·x`.

```
in_W1 (dA) ──┐
              ├─► DSP_1 ──┐
in_H (h_prev)─┘            │
                           ├─► add ─► sat ─► acc/out
in_W2 (dB) ──┐            │
              ├─► DSP_2 ──┘
in_X (x_t) ──┘
```

3 modes:
- **MAC** (linear): chỉ dùng DSP_1, in_W1·in_H accumulate. Tương đương `Unified_PE` MAC.
- **MUL** (element-wise): DSP_1 only, no acc.
- **SSM**: DSP_1 = dA·h, DSP_2 = dB·x, add → h_new. 1 cycle.

**DSP count per PE**: 2. M-cluster 8 lane → **16 DSP**. (vs current 16 lane × 1 DSP = 16 DSP — same!).

**Cycle per t M6 B4** (8-lane M-cluster, lane theo state index):
- 16 state / 8 lane = 2 state passes per channel
- 32 c_grp × 2 state passes × 1 SSM cycle = 64 cycles cho phần h update
- + 32 c_grp × 1 cycle reduction (y accumulate) = 32 cycles
- + setup/transition ~30 cycles
- **Total M6 per t ≈ 130 cycles** (vs ~700 current → **5.4× speedup**)

**Tổng cycle Mamba per t** giảm từ ~1280 → ~600.

**Pros**:
- DSP count unchanged (compared to current Unified_PE 16-lane).
- Native fit cho M6 — chính là contribution paper.
- Reuse MAC mode cho M1A/M1B/M3/M4/M8 (chỉ dùng DSP_1, DSP_2 idle).

**Cons**:
- 1 nửa DSP idle khi MAC mode → DSP utilization drop khi không scan.
- Phức tạp hơn `Unified_PE` (thêm 1 mul + add tree).

### Option PE-C — "SSM-Macro": 1 PE chứa local h_state regfile, 16-wide datapath

**Idea**: Mỗi PE handle **1 state index s** trên tất cả d_inner channel. PE chứa h[s, 0..d_inner-1] regfile bên trong.

```
       ┌──────────────────────────────────────┐
       │  h_regfile[d_inner] (16-bit × 256)   │
       │                                       │
       │   ┌──► DSP_dA ──┐                    │
       │   │              ├─► add ─► sat ──► back to h_regfile
       │   │   ┌► DSP_dB─┘                    │
       │   │   │                               │
       └───┼───┼───────────────────────────────┘
           │   │
       dA, h_prev[c]  dB, x[c]
       (streamed in)
```

- 16 PE × 1 per state, mỗi PE serving entire d_inner channel.
- h_regfile lớn (B4: 256 × 16-bit = 4 Kbit per PE → 16 PE × 4 Kbit = **64 Kbit** total state storage — chấp nhận, dùng LUTRAM hoặc small BRAM).

**DSP count**: 16 PE × 2 DSP = **32 DSP** (cao hơn).

**Cycle per t M6 B4**: 32 c_grp serial × 1 SSM cycle = 32 cycles. **Speedup ~22× vs current.**

**Pros**: cực nhanh M6. State traffic ZERO ra ngoài cluster.

**Cons**:
- DSP 2× so với hiện tại (32 vs 16).
- 16 PE × 4 Kbit regfile khó pack vào single LUTRAM, có thể phải dùng BRAM nhỏ → routing phức tạp.
- M1A/M1B/M8 không tự nhiên fit kiến trúc này (vì những stage đó loop theo channel, không theo state). Phải có second-mode để PE chạy như MAC reduction qua d_in — phức tạp datapath.
- Risk routing congestion vì 16 PE × broadcast dA/dB.

### Option PE-D — "DA-PE + Dedicated h-RegFile cluster-level"

**Idea**: Như PE-B (2 DSP per PE) NHƯNG thêm 1 dedicated h_state regfile ở **cluster level** (ngoài PE) thay vì đẩy qua URAM ram_a.

- M-cluster gồm: 8× PE-B + 1× h_state_regfile (d_inner × d_state × 16-bit, 256×16×16 = 64 Kbit cho B4 — dùng 2-port BRAM_18K).
- h read/write 1 cycle, no URAM round-trip.

**DSP count**: 8 × 2 = **16 DSP**.

**Cycle per t M6 B4** (8-lane lane = state index):
- Loop 32 c_grp × 2 state passes × 1 SSM cycle = 64 cycle
- y reduction 32 c_grp × 1 = 32 cycle
- Saved ~30 cycles per t do bỏ URAM h round-trip
- **Total M6 per t ≈ 100 cycles** (vs ~700 → **7× speedup**)

**Pros**:
- Tương đương PE-B về DSP, nhanh hơn nhờ on-chip state.
- Reuse h_regfile cho cả M6A (update h) và M6B (compute y).

**Cons**:
- 1 BRAM dedicated cho h_state (B4 cần ~64 Kbit = 2 BRAM_18K). Cost nhỏ vs benefit.
- Khi switch giữa Mamba và Inception (ở build dual-cluster), h_regfile unused trong Inception mode — không reusable. Trade-off accept được.

### So sánh tóm tắt

| Option | DSP / cluster | Cycle M6 per t | Resource extra | Verdict |
|--------|---------------|----------------|----------------|---------|
| PE-A "Unified++" | 8 | ~1500 (worse) | 0 | ❌ |
| PE-B "DA-PE 2-DSP" | 16 | ~130 | 0 BRAM | ✅ Tốt |
| PE-C "SSM Macro 16-wide" | 32 | ~30 | 64 Kbit LUTRAM | ⚠️ Quá đắt, lose flexibility |
| **PE-D "DA-PE + h-regfile"** | **16** | **~100** | **2 BRAM** | ✅✅ **Recommended** |

**Chốt: PE-D** — 2-DSP DA-PE per lane, 8-lane M-cluster, dedicated h_state regfile ở cluster level.

---

## 3. Spec PE chốt (PE-D)

### 3.1 Module interface

```verilog
module Mamba_PE (
    input              clk,
    input              rst,

    // Mode select
    input      [2:0]   op_mode,    // MAC / MUL / ADD / SSM / SSM_Y / IDLE
    input              clear_acc,

    // Datapath inputs
    input  signed [15:0] in_W1,    // weight or dA (mul-1 operand A)
    input  signed [15:0] in_H,     // hidden h_prev or activation (mul-1 operand B)
    input  signed [15:0] in_W2,    // dB (mul-2 operand A) — ignored unless SSM
    input  signed [15:0] in_X,     // x_conv (mul-2 operand B) — ignored unless SSM

    // Outputs
    output reg signed [15:0] out_val,    // sat16 result (for h_new / acc / y_partial)
    output reg signed [39:0] acc_raw_dbg // optional acc passthrough for SSM_Y reduction
);
```

### 3.2 Operating modes

| op_mode | Behavior | Inputs used | Latency |
|---------|----------|-------------|---------|
| `MODE_MAC` | acc += in_W1 × in_H; out = sat(acc >> 11) | W1, H | 1 cycle |
| `MODE_MUL` | acc = in_W1 × in_H; out = sat(acc >> 11) | W1, H | 1 cycle |
| `MODE_ADD` | acc = in_W1 + in_H; out = sat(acc) | W1, H | 1 cycle |
| `MODE_SSM` | h_new = sat((W1·H + W2·X) >> 11); out = h_new | W1, H, W2, X | 1 cycle |
| `MODE_SSM_Y` | acc += in_W1 × in_H (giống MAC, dùng cho y_partial += h_new × C) | W1=h_new, H=C | 1 cycle |
| `MODE_IDLE` | clock-gated (Phase 2 add) | — | — |

### 3.3 Datapath sketch

```verilog
wire signed [31:0] m1 = in_W1 * in_H;     // DSP_1
wire signed [31:0] m2 = in_W2 * in_X;     // DSP_2 (only active when SSM)

wire signed [32:0] sum_ssm = {m1[31], m1} + {m2[31], m2};   // 33-bit
wire signed [39:0] sum_ssm_ext = {{7{sum_ssm[32]}}, sum_ssm};

wire signed [39:0] m1_ext = {{8{m1[31]}}, m1};
wire signed [39:0] acc_in = (op_mode == MODE_SSM)   ? sum_ssm_ext :
                            (op_mode == MODE_MUL)   ? m1_ext :
                            (op_mode == MODE_ADD)   ? {{24{in_W1[15]}}, in_W1} + {{24{in_H[15]}}, in_H} :
                            /* MAC, SSM_Y */          (acc_raw + m1_ext);

// Sat + shift on output
wire signed [39:0] acc_next = (clear_acc && op_mode == MODE_MAC) ? m1_ext : acc_in;

always @(posedge clk or posedge rst) begin
    if (rst) begin acc_raw <= 0; out_val <= 0; end
    else begin
        acc_raw <= acc_next;
        out_val <= sat16(acc_next >>> 11);  // SSM/MUL/MAC all shift by FB
    end
end
```

**DSP inference**: synthesizer sẽ infer 2 DSP48E2 per PE — 1 cho m1, 1 cho m2. Adder 33-bit dùng CARRY8.

### 3.4 Timing assumption (giữ 100 MHz @ 10ns)

Critical path expected: DSP_1 (3.2ns) → 33-bit adder (~1.5ns) → 8-bit sign extend (0.2ns) → 40-bit mux (0.8ns) → FF setup (0.5ns) ≈ **6.2ns** < 10ns. Slack ~3.8ns. Có thể đẩy lên 150 MHz về sau nếu cần.

Khi MAC mode: DSP_1 → acc_raw FF feedback (~5ns). Cũng pass.

---

## 4. Cluster (M-Cluster) sketch

8 Mamba_PE + 1 h_state_regfile + 1 reduction tree cho y_partial. Pseudocode top:

```verilog
module M_Cluster (
    input  clk, rst,
    input  [2:0] op_mode,
    input        clear_acc,

    // Per-lane inputs (cluster sequencer feeds these)
    input  [8*16-1:0] in_W1_vec,  // 8 lane × 16-bit
    input  [8*16-1:0] in_H_vec,
    input  [8*16-1:0] in_W2_vec,
    input  [8*16-1:0] in_X_vec,

    // h_regfile interface
    input  [H_ADDR_W-1:0] h_rd_addr,
    input  [H_ADDR_W-1:0] h_wr_addr,
    input                 h_wr_en,
    output [8*16-1:0]     h_rd_data,
    input  [8*16-1:0]     h_wr_data,    // = out_vec when SSM mode

    // Output
    output [8*16-1:0] out_vec,           // 8 lane SSM h_new / MAC partial
    output signed [15:0] y_reduce_out    // sum of 8 lanes (for SSM_Y mode)
);
```

- 8 PE instance, mỗi PE = state index `s_base + lane_idx`.
- `h_state_regfile`: 2-port BRAM 18Kb. Address = `c_grp * d_state/8 + s_pass`. Data = 8 × 16-bit = 128-bit per word.
- Reduction tree: 8-lane adder tree (3-level) cho y_partial reduce theo state lane → 1 lane scalar.

Cycle saving thực sự đến từ **h on-chip 1-cycle read** (vs URAM cascade 4-cycle ngoài) + **fused SSM 1-cycle** (vs 3-cycle hiện tại).

---

## 5. Tích hợp Inception path — pre-thinking

Mục tiêu: A+B trong `plan.md` cần I-cluster chạy SONG SONG M-cluster. Câu hỏi: PE-D có dùng được cho Inception không, hay phải tạo I_PE riêng?

### 5.1 Inception phép tính

Inception trong block: 4 branch (k=1, k=9, k=19, k=39) conv 1D + bottleneck. Mỗi branch là **MAC reduction** trên (k_size × d_inner) per (t, c_out).

Pattern: `y[t, c_out] = sum_{tap=0..k-1, c_in=0..d_inner-1} w[c_out, c_in, tap] × x[t+tap-pad, c_in]`.

Đây **chỉ cần MAC mode** của PE — không cần dual-DSP.

### 5.2 Option I-1: I-cluster dùng PE-D nhưng chỉ MAC mode

- Mỗi I-PE = 1 PE-D, DSP_2 idle suốt Inception phase.
- **Lãng phí 8 DSP** (8 × DSP_2 unused). Total system: 16 (M) + 16 (I) = 32 DSP — cao hơn current 16.
- Pros: code reuse, đồng nhất PE module.

### Option I-2: I-cluster dùng PE riêng "I_PE" 1-DSP (= Unified_PE simplified)

- 8 lane × 1 DSP = 8 DSP cho I-cluster.
- Tổng 8 + 16 = **24 DSP** (chỉ tăng 50% vs current 16, vẫn rất ổn so với 192 DSP có sẵn KV260).
- Slim, clean — Inception không cần SSM mode.

→ **Đề xuất Option I-2**. Tạo 2 PE module riêng: `Mamba_PE` (PE-D) và `Inception_PE` (= Unified_PE simplify). Code duplication ít vì 2 module nhỏ.

### 5.3 Systolic option cho Inception (defer)

Inception k=39 có tap reuse hấp dẫn cho systolic 1D weight-stationary. Có thể later phase 2b thay `Inception_PE` simple thành systolic chain. Phase 2a giữ MAC reduction đơn giản cho song song với M-cluster trước.

---

## 6. Cycle / Resource estimate (rough)

| Metric | Current (16-lane Unified_PE) | M-cluster PE-D (8-lane) + I-cluster I_PE (8-lane) |
|--------|------------------------------|---------------------------------------------------|
| DSP | 16 | 16 (M) + 8 (I) = **24** |
| BRAM (h_regfile) | 0 (h in URAM) | +2 BRAM_18K |
| URAM | 40 (ram_a + ram_b) | ~38 (h removed from ram_a) |
| Cycle Mamba per t (B4) | ~1280 | ~600 |
| Cycle Inception per t (B4) | ~700 (serial after Mamba) | **0** (parallel with Mamba) |
| Total cycle per block B4 (T=250) | ~625K cycle | ~165K cycle → **3.8× throughput** |

(Số trên còn rough; xác nhận khi xong RTL.)

---

## 7. Open questions cần chốt trước RTL

| ID | Câu hỏi | Default đề xuất |
|----|---------|-----------------|
| **P1** | PE-D `op_mode` bit-width 3 hay 4? (4 = thêm slot cho future) | 3 bits (6 mode đủ); expand sau nếu cần |
| **P2** | h_regfile dùng BRAM_18K (cluster level) hay LUTRAM distributed per-lane? | BRAM (đơn giản, 1 port write 1 port read) |
| **P3** | y_reduction tree: kết hợp vào M_Cluster hay làm module riêng `Reduce8`? | Module riêng — reuse được cho M6B y compute |
| **P4** | M-cluster lane = state index s hay channel index c? | **state index** — match h_regfile addressing pattern |
| **P5** | DSP_2 idle khi MAC mode — có gate hay để Vivado tự optimize? | Để Vivado tự (DSP có CE pin) |
| **P6** | RMSNorm vẫn dùng `RMSNorm_Mul` module hiện có hay tích hợp vào M_Cluster? | Giữ riêng — interface ổn |
| **P7** | clear_acc semantics khi SSM mode? | Ignored — SSM always overwrite acc với h_new |
| **P8** | Có cần "MODE_MAC_VEC" cho I-cluster broadcast pe_A khác lane? | I_PE riêng — không cần ở Mamba_PE |

---

## 8. Roadmap implementation PE → M-Cluster → Top

**Step 1** (1-2 ngày): Viết `Mamba_PE.v` + unit-level testbench. Verify 6 mode dùng synthetic vector. Compare bit-exact với reference Python.

**Step 2** (1 ngày): Viết `Reduce8.v` (8-lane signed adder tree, 16-bit→20-bit pipelined 1 stage).

**Step 3** (2-3 ngày): Viết `H_RegFile.v` (BRAM 2-port wrapper, 8×16-bit word, ADDR width = ceil(log2(d_inner × d_state / 8))) + `M_Cluster.v` wrap.

**Step 4** (3-5 ngày): Viết `FSM_M.v` driving cluster. State machine: M1A → M1B → RMSNorm → M2 → M3 → M4 → M5 → M6 (init_h, da/db prefetch, ssm loop, y reduce) → M7 → M8. Mỗi sub-state đơn giản hơn `ITM_CONTROLLER_v3` vì PE đã fused SSM.

**Step 5** (2 ngày): `Mamba_Dedicated_Top.v` cho build #2 (standalone). TB synthetic random. Measure cycle.

**Step 6** (parallel với step 5, 2 ngày): Viết `Inception_PE.v` (= Unified_PE simplified, ADD mode optional) + sketch `I_Cluster.v` outline.

**Decision gate trước step 4**: synth `Mamba_PE` + `M_Cluster` standalone qua OOC để confirm DSP count = 16, timing ≥ 100 MHz, BRAM = 2. Nếu không đạt → revise PE-D.

---

## 9. Đóng

Plan này tập trung Mamba PE, đủ context để bắt đầu viết RTL `Mamba_PE.v`. Inception đã thấy đường để chạy song song (Option I-2 với `Inception_PE` riêng).

**Next concrete step**: viết `Mamba_PE.v` skeleton + chốt P1-P8. Sau khi user OK PE-D + chốt open questions → tôi tạo file.
