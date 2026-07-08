# ITMN Accelerator — Định hướng Novelty & Workflow Báo cáo

*Last edited: 2026-06-23 — K1-K8 chốt v1*

Tài liệu này phục vụ 2 mục tiêu user nêu:

1. **Đánh giá lại** thiết kế RTL hiện tại có thực sự "novel" hay chỉ là một functional baseline; chỉ ra điểm chưa novel và đề xuất các hướng đột phá đủ tầm paper sinh viên (wearable / edge).
2. **Đề xuất workflow hoàn chỉnh** để báo cáo ITMN nhưng vẫn so sánh fair với hai họ paper khác (Mamba-HW only và Inception/CNN-HW only).

Không phải plan implementation cụ thể đến từng line — đó là việc của các CP-x sau khi đã chốt hướng.

---

## 0. Quyết định đã chốt (2026-06-23)

| ID | Câu hỏi | Chốt |
|----|---------|------|
| K1 | Hướng chính | **A + B** (Dual-Path Cluster + Reconfigurable Mode) |
| K2 | Mixed-precision C | Bỏ — giữ INT16 (Q4.11) xuyên suốt |
| K3 | Parallel scan D | Bỏ |
| K4 | RTL build strategy | **3 RTL build hoàn toàn riêng** (ITMN-Dual / Mamba-only / Inception-only) |
| K5 | Pytorch training & accuracy verify | **Không train, không verify accuracy** — synthetic random data, đo thuần HW (throughput / resource / timing / power) |
| K6 | Target FPGA | KV260 (xck26-sfvc784-2LV-c) |
| K7 | Fmax | Từ từ — đúng kiến trúc + đúng số liệu trước, tăng Fmax sau |
| K8 | Encoder/GAP/FC precision | INT16 (Q4.11) |

**Hệ quả lớn của K5**:

- Toàn bộ chain `extract_itm_full.py` → `golden_all/*.txt` → `verify_byte_exact.py` **không dùng** cho 3 build mới. Vẫn giữ chain hiện tại cho ITMN gốc làm reference functional sanity check, nhưng KHÔNG là gate cho 3 build mới.
- TB cho 3 build mới chỉ cần: random input 16-bit signed, dummy weight init bằng `$urandom`, đếm cycle từ `start` → `done`. Output không so golden.
- Không claim được "ITMN AUC tốt hơn baseline" trong paper. Claim duy nhất là **HW-side**: dual-path saves cycle, dedicated Mamba-only / Inception-only ăn ít resource hơn full. Đây là một trade-off scope hợp lý cho student paper khi không có ngân sách training.
- Pytorch model hiện tại vẫn dùng cho 1 việc duy nhất: extract **shape, dimension, weight tensor size** để chia memory map. Không gen golden mới.

---

## 1. Đánh giá thẳng thắn thiết kế hiện tại

### 1.1 Trạng thái

| Item | Số liệu |
|------|---------|
| AUC float / Q4.11 (legacy reference) | 0.9354 / 0.9328 |
| Fmax | 100 MHz (slack +0.646 ns @ 10 ns) |
| Resource (full ITM, post-RAM-2/3/4) | 10491 LUT / 4504 FF / 59 DSP / 118 BRAM / 40 URAM |
| D1 OOC: Mamba_Top | 7513 LUT, 39 DSP, slack +0.71 ns |
| D1 OOC: Inception_Top | 5346 LUT, 37 DSP, slack +1.97 ns |
| Throughput | 1.66 inf/s (sau CP-1 RMSNorm pipeline) |
| Scope | 5 ITM block + cascade pool. Encoder/GAP/FC còn ở host Python. |

### 1.2 Vì sao thiết kế hiện tại CHƯA novel

Đây là phần khó nghe — viết để tự đối diện:

1. **PE Array là một MAC engine generic.** `Unified_PE` = 1 DSP48E2 + 40-bit acc + sat16, hỗ trợ MAC/MUL/ADD. Không có một quyết định kiến trúc nào *xuất phát từ đặc thù* của Mamba SSM hay Inception. Cùng một PE chạy: P1 1×1 conv, Inception k=9/19/39, M1 input projection, M2 depthwise conv, M3 x_proj, M4 dt_proj, M5 discretize, M6 SSM scan, M7 gating, M8 out_proj. Mọi thứ bị **flatten** thành "scalar broadcast hoặc vector × vector". Đây là điểm yếu paper-wise: reviewer sẽ hỏi "what makes this an *ITMN* accelerator vs a generic MAC array?" — hiện tại câu trả lời là "FSM controller".

2. **Inception và Mamba bị serialize trong controller, dù model Pytorch chạy song song.** Trong `ITMBlock.forward`: `x1 = inception(x); x2 = mamba(x); out = x1 + x2`. Hai branch độc lập, có thể chạy đồng thời. RTL hiện tại serialize P1 → Inception → Mamba → FIN. Bỏ phí ~50% data-level parallelism inherent của model. Đây là **cơ hội novelty bị bỏ lỡ** — chính cái "hybrid local+global" của paper ITMN sẽ ánh xạ tự nhiên sang "dual-pipeline parallel HW", không ai đang làm điều đó.

3. **Inception 4 branch (k=1, k=9, k=19, k=39) bị serialize trên cùng PE_Array.** Mỗi branch chạy hết rồi mới đến branch kế. Trade-off "tiết kiệm PE" nhưng critical-path không phải hiện tại nằm ở đó — nên không cần share. Đặc biệt k=9/19/39 có **rất khác về reuse pattern** (k=39 reuse weight 39 lần qua cùng tap, k=9 chỉ 9 lần) → mỗi kernel size hợp với một micro-architecture khác (k=39 nên dùng systolic 1D weight-stationary, k=1 nên dùng dense MAC). Đang ép cùng một datapath là sub-optimal cả Fmax lẫn DSP.

4. **Adder rải rác làm "function" tổ hợp ở controller.** `sat_add16`, 17-bit add trong `bn_relu`, 16-input adder tree `norm_sq16_fn`, 16-lane `max_buf` compare. Đây là "đặt đại vào controller cho tiện" chứ không có quan điểm kiến trúc.

5. **Mamba SSM bị ép vào MAC.** Recurrence `h_t = Ā·h_{t-1} + B̄·x_t` về bản chất là state update tuần tự theo `t`. Mamba HW trong literature thường tận dụng (a) A diagonal sau parameterization → 16 state-dim song song; (b) M3/M4/M5 (x_proj/dt_proj/discretize) độc lập với scan và overlap được. RTL hiện tại loop naive qua PE Array.

6. **D1 standalone không phải "Mamba-only design" — chỉ là "full design với phần Inception bị define-cut".** Vivado vẫn instantiate đầy đủ Memory_System / Const_Storage / PE_Array. URAM, BRAM unchanged giữa 3 build. Reviewer paper khắt khe có thể bác: "so sánh fair phải là Mamba accelerator được thiết kế riêng từ đầu, không phải bản gốc cắt nửa". → Đây chính là lý do K4 chọn 3 RTL build riêng.

7. **Wearable angle đang yếu.** Paper claim wearable nhưng RTL chưa có streaming I/O, power gating, sparsity. (Đã chốt KHÔNG làm — chấp nhận điểm yếu này trong scope hiện tại.)

### 1.3 Điểm thiết kế *đã làm đúng* (giữ lại)

- Q4.11 quantization với byte-exact verification chain (cho ITMN gốc — reference).
- Const_Storage consolidation (D3.A).
- Compact memory map URAM downsizing (40 URAM vs 64).
- RMSNorm_Mul pipeline (CP-1).
- Tách Memory_System khỏi controller.
- D1 OOC scripts (`Mamba_OOC.tcl`, `Inception_OOC.tcl`, `D1_Compare.tcl`) — reuse được cho 3 build mới.

---

## 2. Hướng kiến trúc đã chốt: A + B

### Direction A — Dual-Path Parallel Cluster

**Idea**: Tách PE_Array 16 lane thành 2 cluster nhỏ chạy song song trong cùng block:

- **I-cluster** (8 lane, Inception-specialized): tối ưu cho conv k=1/9/19/39. Đề xuất weight-stationary 1D systolic với line buffer theo trục thời gian. Một option: chia k=39 thành 4 sub-window 10-tap để giảm DSP nếu cần.
- **M-cluster** (8 lane, Mamba-specialized): tối ưu cho SSM diagonal recurrence + depthwise conv k=4. Có dedicated `h_state` register file (state_dim × 16-bit) tách khỏi RAM để bỏ qua URAM 1-cycle latency trong scan loop.

FSM controller chia thành 2 sub-FSM chạy đồng thời, đồng bộ ở FIN stage (BN + ReLU combine `inc + mam`).

**Novelty claim**: First HW accelerator matching the dual-pathway structure of hybrid Inception+SSM models — chứng minh qua throughput vs baseline serial design.

**Risk chính**:

- Memory contention: cả 2 cluster đều cần đọc `p1_out`. Giải pháp: duplicate region p1_out vào cả ram_a và ram_b (cost +1-2 URAM).
- FSM phức tạp: 2 FSM + handshake ở FIN. Đề xuất explicit `done_inception` + `done_mamba` flag, FIN wait until both.
- DSP có thể tăng nếu I-cluster systolic dùng nhiều multiplier. Estimate: tổng DSP ~ 60-80 (vs 59 hiện tại), chấp nhận được nếu cycle giảm 40%+.

### Direction B — Reconfigurable Mode Switch

**Idea**: Trong build **ITMN_Dual** (build #1 ở §3), thêm register `mode[1:0]`:

- `mode=2'b00` → full ITMN: I-cluster + M-cluster đồng thời, FIN combine.
- `mode=2'b01` → Mamba-only path: bypass I-cluster, M-cluster only, FIN output = relu(mamba).
- `mode=2'b10` → Inception-only path: bypass M-cluster, I-cluster only, FIN output = bn_relu(inception).

Mục đích: cho phép cùng 1 bitstream ITMN_Dual chạy 3 workload khác nhau → đo throughput/cycle 3 mode dễ dàng, không cần reprogram.

**Lưu ý**: B chỉ áp dụng cho build #1 (ITMN_Dual). Build #2 (Mamba_Dedicated) và #3 (Inception_Dedicated) ở §3 KHÔNG có mode register — chúng là dedicated standalone designs để fair-compare resource với paper bên ngoài.

---

## 3. Workflow so sánh ITMN vs Mamba vs Inception (K4 = 3 build riêng, K5 = synthetic data)

### 3.1 Vấn đề

Paper Mamba-HW so với bạn ở Mamba-only. Paper Inception/CNN-HW so với bạn ở Inception-only. Bạn cần báo cáo 3 con số riêng, FAIR, trên cùng FPGA target (KV260), cùng precision (Q4.11). Vì K5 đã chốt không verify accuracy, claim của paper sẽ thuần **HW efficiency** (throughput, resource, energy), không bao gồm AUC.

### 3.2 Sơ đồ benchmark đã chốt

```
                  ┌──── Synthetic data only (K5) ───────────┐
                  │  random Q4.11 input + dummy weights     │
                  │  (no Pytorch training, no AUC verify)   │
                  └────────────┬────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  Build #1                Build #2                Build #3
  ITMN_Dual               Mamba_Dedicated         Inception_Dedicated
  (A+B, 3 mode)           (dedicated HW          (dedicated HW
                          for Mamba only)         for Inception only)
        │                      │                      │
   codebase A             codebase B             codebase C
   sources_v3/itmn_dual   sources_v3/mamba       sources_v3/inception
        │                      │                      │
        ▼                      ▼                      ▼
   Vivado synth/impl       Vivado synth/impl      Vivado synth/impl
   + power report          + power report         + power report
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                    tools/parse_reports.py
                               │
                               ▼
           CSV → matplotlib bar/scatter → LaTeX table
```

### 3.3 3 codebase RTL (K4)

Đề xuất layout repo:

```
ITMN_RTL_srcs/
  sources_1/new/         # GIỮ NGUYÊN — full ITMN serial baseline (D1 OOC vẫn chạy được)
  sources_v3/
    common/              # shared (BRAM_256b, Const_Storage skeleton, RSqrt_ROM, LUTs)
    itmn_dual/           # Build #1 — A+B, dual-path cluster + mode register
      ITMN_Dual_Top.v
      I_Cluster.v
      M_Cluster.v
      FSM_I.v
      FSM_M.v
      Mode_Switcher.v
    mamba/               # Build #2 — Mamba dedicated (M-cluster + minimal infra)
      Mamba_Dedicated_Top.v
      M_Cluster.v        # symlink hoặc copy
      FSM_M.v
    inception/           # Build #3 — Inception dedicated (I-cluster + minimal infra)
      Inception_Dedicated_Top.v
      I_Cluster.v
      FSM_I.v
  sim_1/new/
    tb_itmn_dual.v       # synthetic random, đo cycle
    tb_mamba.v
    tb_inception.v
```

Rationale chia thế:

- `common/`: BRAM/LUT primitives nên share để tránh duplicate code khi sửa.
- `I_Cluster.v` và `M_Cluster.v`: viết MỘT lần ở `itmn_dual/`, instance lại ở `mamba/` và `inception/`. Khi dedicated build dùng cluster riêng lẻ, FSM trên đó sẽ feed input/output port tương ứng, không cần dual handshake.
- 3 top-level module riêng → Vivado mỗi build chỉ thấy 1 top → resource report fair, không bị "bloat" do logic không dùng.

### 3.4 Testbench với synthetic data (K5)

Mỗi TB:

1. Reset.
2. `$urandom_range` fill input buffer (T × d_in × 16-bit) với random Q4.11 trong range ±2.0 → tránh saturate trivial.
3. `$urandom` fill weight buffer (P1/Bot/B1-B4/M_x/M_z/conv1d/x_proj/dt_proj/A_log/D_param/out_proj) trong range ±0.5.
4. Pulse `start`.
5. Monitor `done` + per-phase done flag (`done_phase1`, `done_inception`, `done_mamba`, `done_fin`).
6. Print cycle counts.
7. KHÔNG dump output, KHÔNG so golden.

Files cần:

- `tb_itmn_dual.v`: chạy full 5 block + cascade. Print cycle theo mode (full / mamba / inception).
- `tb_mamba.v`: 1 inference Mamba-only block, có loop count (multiple block test).
- `tb_inception.v`: tương tự, Inception-only.

Reuse Memory_System DMA path để load random data (đỡ phải viết wrapper mới).

### 3.5 Metrics thu thập per build (K5 → bỏ AUC)

| Metric | Source | Note |
|--------|--------|------|
| LUT, FF, DSP, BRAM, URAM | `report_utilization.rpt` | Per-hierarchy + total |
| Fmax (achieved) | `report_timing_summary.rpt` | WNS @ XDC period 10 ns |
| Power: static + dynamic | `report_power.rpt` | Cần SAIF từ post-impl sim với synthetic data; nếu skip SAIF dùng vectorless |
| Cycle / inference | TB simulation log | Per phase, per block |
| Throughput (inf/s) | Fmax × T_total | Tính từ cycle |
| Energy / inference (μJ) | Power_dynamic × latency | Wearable framing |
| ~~AUC, TPR, F1~~ | ~~Pytorch~~ | **BỎ — K5** |
| GOPS | Compute từ MAC count × Fmax / cycle | Cho normalize so paper khác |

### 3.6 So sánh với paper bên ngoài

Pick 6-8 paper. Vì không có AUC riêng, focus comparison thuần HW metric:

| Loại paper | Vai trò bench | Metric so |
|------------|--------------|-----------|
| Mamba-HW FPGA (NLP/Vision, 2024-2025) | Mamba-only fair | LUT/DSP/Fmax/throughput/GOPS-per-LUT |
| InceptionTime-HW or 1D-CNN ECG accel | Inception-only fair | Same |
| (Optional) MVMS / ATI-CNN / ECGNet model-only papers | Algorithm reference cho framing introduction, không so HW | — |

Normalize FPGA family: Zynq-7000 LUT_eqv ≠ UltraScale+ LUT_eqv. Giữ 2 dạng bảng: raw + normalized-per-resource.

### 3.7 Visualization

3 hình cho paper / báo cáo:

1. **Resource bar chart**: 3 build × {LUT, DSP, BRAM, URAM} — chứng minh ITMN_Dual KHÔNG = Mamba_Dedicated + Inception_Dedicated (cluster cộng < tổng vì share memory). Demonstrate "hybrid HW efficiency".
2. **Cycle breakdown stacked bar**: 3 build × {P1, Inception, Mamba, FIN, cascade} per block — chứng minh dual-path saves cycle.
3. **Throughput vs Resource scatter (Pareto)**: 3 build của bạn + 6-8 paper external. Position dot ITMN_Dual trên Pareto frontier.

Tool: `matplotlib`, script `tools/plot_compare.py`.

---

## 4. Roadmap cụ thể (timeline 8-10 tuần, đã giảm scope do bỏ K5)

### Phase 0 (tuần 1) — Setup repo + synthetic TB infra

- [ ] Tạo `ITMN_RTL_srcs/sources_v3/` layout (common, itmn_dual, mamba, inception).
- [ ] Copy primitives BRAM_256b, Const_Storage, LUT_s từ `sources_1/new/` sang `common/`.
- [ ] Viết skeleton 3 TB synthetic random (chưa cần module thật — TB chỉ wire ports + cycle counter).
- [ ] Setup 3 TCL OOC script: `ITMN_Dual_OOC.tcl`, `Mamba_Dedicated_OOC.tcl`, `Inception_Dedicated_OOC.tcl` (clone từ `Mamba_OOC.tcl` hiện tại).

### Phase 1 (tuần 2) — End-to-end stages: encoder + GAP + FC

Để 3 build comparable, cycle count phải bao gồm hết end-to-end. Theo OPTIMIZATION_NOTES §9 plan cũ:

- [ ] Encoder Conv1D 12→64 (mode INT16). Tích hợp vào FSM của ITMN_Dual_Top trước S_P1.
- [ ] GAP sau block 4.
- [ ] FC layer (d_in=128, num_classes=5 cho PTB-XL super).
- [ ] Cùng 3 stage này được copy vào Mamba_Dedicated_Top và Inception_Dedicated_Top.

Lưu ý: vì K5 không verify accuracy → KHÔNG cần load weight thật. Random init là đủ. Chỉ cần đúng dimension để cycle/resource đúng.

### Phase 2 (tuần 3-5) — I-Cluster + M-Cluster module (Direction A)

**Phase 2a — M-Cluster** (tuần 3-4):

- [ ] Thiết kế micro-arch: 8-lane datapath, dedicated `h_state_regfile` (16 × 16-bit cho B0-B3, 16 × 16-bit cho B4).
- [ ] M1-M8 sub-states FSM_M.
- [ ] Diagonal-A multiplier: `h_t[n] = ā[n] * h_{t-1}[n] + b̄_t[n] * x_t`. 8 lane parallel, lane n giữ state index n.
- [ ] Depthwise conv k=4 dùng line buffer trên-die thay vì URAM round-trip.
- [ ] Standalone tb_mamba.v synth + impl + cycle measure.

**Phase 2b — I-Cluster** (tuần 4-5):

- [ ] Thiết kế micro-arch: 8-lane systolic 1D weight-stationary cho k=9/19/39. Bottleneck k=1 dùng dense MAC.
- [ ] BR sub-states FSM_I.
- [ ] Line buffer cho conv tap reuse (giảm read RAM).
- [ ] Standalone tb_inception.v synth + impl + cycle measure.

**Phase 2c — Memory contention resolution** (tuần 5):

- [ ] Quyết định duplicate p1_out hay dual-port URAM (đo lại với cả 2 option).

### Phase 3 (tuần 6) — ITMN_Dual integration + Mode register (Direction B)

- [ ] Top-level ITMN_Dual_Top instantiate cả I + M cluster + Mode_Switcher.
- [ ] FIN combine: 16-lane adder + bn_relu, wait `done_inception && done_mamba`.
- [ ] Mode register: `mode[1:0]` chọn FIN source.
- [ ] tb_itmn_dual.v chạy 3 mode, in 3 cycle count.

### Phase 4 (tuần 7) — Benchmarking infrastructure

- [ ] `tools/run_3builds.sh`: chạy 3 Vivado synth + impl + report sequential (overnight).
- [ ] `tools/parse_reports.py`: extract LUT/FF/DSP/BRAM/URAM/Fmax/power → CSV.
- [ ] `tools/measure_cycles.py`: parse xsim log → cycle CSV.
- [ ] `tools/plot_compare.py`: 3 figure ở §3.7.

### Phase 5 (tuần 8-9) — Paper writing & visualization

- [ ] Architecture diagram (3 build).
- [ ] Tables: §3.5 metrics, comparison với 6-8 paper.
- [ ] Figures: resource bar, cycle breakdown, Pareto scatter.
- [ ] Ablation: ITMN_Dual full vs mode=mamba vs mode=inception (cùng bitstream) — chứng minh mode register hoạt động.
- [ ] Discussion: hybrid HW efficiency = Mamba + Inception − shared memory.

### Phase 6 (tuần 10) — Buffer / advisor revision

---

## 5. Câu hỏi mở cần chốt khi vào từng phase

| ID | Câu hỏi | Khi nào quyết |
|----|---------|---------------|
| Q5.1 | M-Cluster `h_state_regfile` dùng FF distributed hay LUTRAM? | Phase 2a — depend critical path |
| Q5.2 | I-Cluster systolic chiều spatial (channel) hay temporal (time)? | Phase 2b — depend k=39 tap reuse |
| Q5.3 | Cascade pool giữa B1-B2 và B3-B4: làm trong cluster nào? | Phase 3 — dual-path nên có cascade ở FIN hoặc ngoài |
| Q5.4 | Random seed cho synthetic TB: fixed (reproducible) hay variable? | Phase 0 — fixed cho consistent cycle |
| Q5.5 | Power report: dùng SAIF (chính xác hơn, chậm) hay vectorless (nhanh, gần đúng)? | Phase 4 — chọn 1 cho consistency |
| Q5.6 | Paper external benchmark: ưu tiên Mamba-HW NLP/Vision (đa số) hay strict ECG-only? | Phase 5 — depend reviewer expectation |

---

## 6. Rủi ro & cách giảm thiểu (cập nhật theo K5)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dual-path memory contention | Cycle saving thấp hơn expected | Duplicate `p1_out` region — cost 1 URAM extra |
| 8-lane I-cluster không đủ DSP cho k=39 | Throughput giảm | Time-tile k=39 thành 4 sub-window × 10-tap |
| Vivado synth time × 3 build | Iterate chậm | Overnight CI, ưu tiên xsim cho cycle measure |
| Paper Mamba-HW khó tìm trên FPGA ECG domain | Khó fair compare | Mở rộng tới Mamba-HW general (NLP/Vision), disclaimer |
| Không verify accuracy → reviewer hỏi "có chắc thiết kế đúng?" | Credibility | Giữ chain byte-exact cho ITMN gốc (legacy build) làm functional proof; 3 build mới reuse cùng arithmetic primitives (PE_Array INT16 Q4.11) — nếu primitives đúng thì FSM mới cũng đúng. Note rõ trong paper. |
| Synthetic random có thể không hit worst-case timing/power | Số liệu hơi optimistic | Pick worst-case seed; có thể chạy nhiều seed average |

---

## 7. Đóng plan

Plan này là "chiến lược cấp cao". Các CP-x cụ thể (CP-5 M-cluster diag-SSM, CP-6 I-cluster systolic, CP-7 dual-path memory) sẽ được elab ở `FMAX_RESOURCE_PLAN.md` mới khi vào từng Phase.

**Next concrete step**: bắt đầu **Phase 0 — setup repo `sources_v3/` + 3 TB skeleton synthetic**. Tôi có thể tạo skeleton files khi bạn xác nhận layout `sources_v3/` ở §3.3 chấp nhận được.
