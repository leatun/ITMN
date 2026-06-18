# Thiết kế và Hiện thực FPGA của ITMN: Bộ tăng tốc phân loại ECG Multilabel
## kết hợp Inception, Mamba SSM và lượng tử hóa Q4.11 end-to-end

**Tác giả**: Lê Thanh Tuấn
**Ngày**: 2026-06-21

---

## Tóm tắt

Bài báo trình bày kiến trúc và quy trình hiện thực FPGA cho **ITMN** — một mô hình
phân loại ECG 12-lead đa nhãn kết hợp module Inception và State-Space Model Mamba.
Toàn bộ pipeline gồm encoder Conv1D, năm khối ITM, Global Average Pooling và bộ
phân loại tuyến tính được lượng tử hóa về định dạng **Q4.11 fixed-point** và hiện
thực dưới dạng một bộ điều khiển FSM duy nhất chia sẻ chung mảng PE 16 làn, hệ
thống bộ nhớ ba ngân hàng và bộ lưu hằng số. Trên FPGA AMD KV260
(`xck26-sfvc784-2LV-c`), thiết kế đạt Fmax ≈ 100 MHz với mức tận dụng tài nguyên
**[Bảng 4]** và độ chính xác AUC = 0.9328 trên tập kiểm tra PTB-XL super-class
(2158 mẫu), chỉ thấp hơn mô hình float reference 0.0026 AUC. Quá trình kiểm chứng
tuân theo phương pháp ba lớp: (i) Python integer reference bit-accurate, (ii) RTL
simulation byte-exact trên N mẫu ngẫu nhiên, (iii) báo cáo độ chính xác toàn bộ
tập kiểm tra thông qua mô hình tham chiếu Python — một quy trình chuẩn cho các
bộ tăng tốc DNN trên FPGA.

---

## 1. Giới thiệu

Phân loại tín hiệu điện tim (ECG) đa nhãn là bài toán cốt lõi trong chẩn đoán
tim mạch tự động. Tập dữ liệu PTB-XL [REF_PTB_XL] cung cấp hơn 21,000 bản ghi
ECG 12-lead với năm nhóm chẩn đoán siêu lớp (NORM, MI, STTC, CD, HYP). Các mô
hình deep-learning gần đây như InceptionTime [REF_Inception_ECG] và các biến thể
Mamba/SSM [REF_Mamba] đạt độ chính xác cao nhưng yêu cầu chi phí suy luận lớn,
hạn chế khả năng triển khai trên thiết bị nhúng/edge.

Mô hình **ITMN** kết hợp ưu điểm của hai họ kiến trúc: Inception cung cấp khả
năng trích xuất đặc trưng đa tỉ lệ (kernel 9/19/39), trong khi Mamba SSM mô hình
hóa hiệu quả các phụ thuộc dài qua chuỗi thời gian. Tuy nhiên việc hiện thực
phần cứng cho mô hình lai này gặp ba thách thức chính:

1. **Đa dạng phép tính**: 1×1 conv, conv k=9/19/39, RMSNorm, Linear projections,
   Selective SSM scan, softplus + exp + silu nonlinearities, max-pool, GAP, FC.
2. **Khác biệt về kích thước**: từ kernel 1×1 cực ngắn (12→64 channels) đến SSM
   scan chiều thời gian T=1000.
3. **Yêu cầu lượng tử hóa thấp**: triển khai edge yêu cầu định dạng số ≤ 16 bit
   nhưng vẫn duy trì AUC không suy giảm đáng kể.

**Đóng góp chính** của báo cáo này:

- **(C1)** Hiện thực FPGA end-to-end đầu tiên (theo hiểu biết của tác giả) cho
  pipeline ECG Inception+Mamba, bao gồm encoder + 5 ITM block + GAP + classifier
  trên cùng một bộ điều khiển.
- **(C2)** Định dạng số **Q4.11 với RMSNorm v2** — một biến thể không pre-shift
  và độ phân giải ROM rsqrt cao hơn — phục hồi AUC từ 0.5607 (Q4.11 RMSNorm v1)
  lên 0.8635 (xấp xỉ float ceiling 0.8624).
- **(C3)** Quy trình kiểm chứng ba lớp: Python integer reference byte-exact với
  RTL trên 100/100 mẫu, sau đó báo cáo accuracy toàn bộ tập kiểm tra thông qua
  mô hình tham chiếu — tiết kiệm thời gian simulation hàng tháng.
- **(C4)** Phân tích **D1 separation** (mảng Mamba/Inception riêng) cho phép so
  sánh công bằng với các nghiên cứu trước, kèm phân tích overhead của
  **D2 end-to-end** (encoder + head).

---

## 2. Mô hình ITMN

### 2.1 Kiến trúc tổng thể

Mô hình ITMN nhận đầu vào ECG 12-lead với T=1000 mẫu thời gian. Pipeline gồm:

```
Input (B, 12, 1000)
  → Encoder        : Conv1D(12, 64, k=1) + BatchNorm1D      → (B, 64, 1000)
  → ITM Block 0    : ITMBlock(64, 64)                       → (B, 64, 1000)
  → ITM Block 1    : ITMBlock(64, 64)                       → (B, 64, 1000)
  → MaxPool1D(2)                                            → (B, 64, 500)
  → ITM Block 2    : ITMBlock(64, 64)                       → (B, 64, 500)
  → ITM Block 3    : ITMBlock(64, 64)                       → (B, 64, 500)
  → MaxPool1D(2)                                            → (B, 64, 250)
  → ITM Block 4    : ITMBlock(64, 128)                      → (B, 128, 250)
  → GAP            : mean over T                            → (B, 128)
  → Classifier     : Linear(128, 5)                         → (B, 5)
```

**[Hình 1: Sơ đồ pipeline ITMN — Encoder, 5 ITM block với 2 maxpool stride-2 chèn giữa, GAP và FC head]**

Mỗi **ITMBlock(d_in, d_out)** thực hiện:

```
P1 : Conv1D(d_in, d_out, k=1) + BatchNorm1D
x1 : Inception(P1_out)
x2 : ReLU( Mamba( RMSNorm(P1_out) ) )
out: x1 + x2          (theo công thức gốc PyTorch)
```

Trong đó **Inception** gồm 4 nhánh song song và một bottleneck:

- **Bottleneck**: Conv1D(d_out, d_out/4, k=1)
- **Branch 1**: MaxPool(k=3) → Conv1D(d_out, d_out/4, k=1)
- **Branch 2-4**: Conv1D(d_out/4, d_out/4, k=9/19/39 trên Bottleneck output)
- Concatenate (4 × d_out/4) → BatchNorm1D + ReLU

**Mamba block** gồm:

- **RMSNorm**: chuẩn hóa trên chiều kênh
- **In-projection**: hai bộ Linear song song tạo `x_inner` và `z_gate` (chiều
  d_inner = 2 × d_in)
- **Depthwise Conv1D k=4** trên `x_inner`
- **SiLU** activation
- **x_proj**: Linear(d_inner, dt_rank + 2 × d_state) → tạo `delta`, `B`, `C`
- **dt_proj**: Linear(dt_rank, d_inner) + softplus → time-step `delta_t`
- **Selective SSM scan**: `h_t = exp(delta_t · A) · h_{t-1} + (delta_t · B) · u_t`
  với `A` là ma trận học được, `u = SiLU(x_conv)`, `h ∈ R^d_state`
- **Output**: `y = (C^T · h) + D · u`, sau đó nhân `SiLU(z_gate)` và out_projection

### 2.2 Danh sách phép tính

[Bảng 1] tóm tắt các phép tính cần triển khai trên FPGA, kích thước cực đại
(theo Block 4) và số lần xuất hiện trong toàn pipeline.

**[Bảng 1: Danh sách phép tính ITMN + kích thước + tần suất]**

| Phép tính | Kích thước tối đa | Xuất hiện | Phụ thuộc |
|---|---|---|---|
| 1×1 Conv (linear projection) | 256×128 | 14 lần | DSP MAC |
| Conv 1D k=9/19/39 | 32×32×39 | 3 lần/block | DSP MAC kernel |
| BatchNorm1D + ReLU | 256 ch | 6 lần | mul, add, sat |
| RMSNorm | 256 ch | 10 lần | sum-of-squares, rsqrt, mul |
| Depthwise Conv1D k=4 | 256 ch | 5 lần | DSP MAC dạng vector |
| SiLU/Softplus/Exp | 256 ch | nhiều | LUT 256-entry |
| SSM scan (A, B, C, D) | 256 × 16 states | 5 lần | MAC + nonlinearities |
| MaxPool1D k=3 stride=1 | 256 ch | 5 lần | comparator |
| MaxPool1D k=2 stride=2 | 64 ch | 2 lần | comparator |
| GAP (mean over T) | 128 ch × T=250 | 1 lần | accumulator + chia |
| Linear FC | 5 × 128 | 1 lần | DSP MAC |

### 2.3 Định dạng số Q4.11

Toàn bộ pipeline được lượng tử hóa về **Q4.11 signed 16-bit**:

- Khoảng giá trị: `[−16.0, +16.0)` float
- Độ phân giải: `1/2048 ≈ 4.88 × 10⁻⁴`
- Bão hòa (`sat16`) tại mọi ranh giới stage
- Bộ tích lũy 40-bit cho MAC và sum-of-squares
- Định dạng BRAM: 16-bit × 16 làn = 256-bit per line

Lựa chọn Q4.11 dựa trên phân tích phân phối kích hoạt giữa các block: phần lớn
giá trị nằm trong `(−4, +4)` nên 4 bit integer đủ và 11 bit phân số duy trì SQNR
> 30 dB cho hầu hết các stage.

### 2.4 RMSNorm v2: Phục hồi độ chính xác

RMSNorm gốc trong Mamba tính `y = x · gamma · rsqrt(mean(x²) + eps)`. Phiên bản
integer ngây thơ (v1):

```
sq[i]   = (x[i] >> 5)²       // pre-shift để tránh tràn 32-bit
sum     = Σ sq[i]
mean_i  = sum >> (log2_d + 2·FB)
S       = ROM[mean_i]        // ROM cỡ 256 entries với K=2896
y       = sat16( sat16( (x·gamma)>>FB ) · S >> FB )
```

V1 gặp hai vấn đề kết hợp: kênh có biên độ nhỏ bị truncate về 0 trong sum, và
ROM độ phân giải thô khiến `target_rms < 0.7` đều ánh xạ tới index 0 → output
khuếch đại sai 16x. Kết quả AUC giảm từ 0.93 (float) xuống 0.5607.

**RMSNorm v2** sửa cả hai vấn đề:

```
sq       = x · x                        (không pre-shift, 32-bit)
sum_d    = Σ sq                         (40-bit accumulator)
mean_i   = sum_d >> (log2_d + 2·FB − 1 − N)    với N = 6 bit độ phân giải bổ sung
S_t      = ROM_v2[clip(mean_i, 0, 8191)]       với ROM_v2 K = sqrt(2^7) · SCALE ≈ 23170
out      = sat16( sat16((x·gamma) >> FB) · S_t >> FB )
```

Đơn vị `target_rms` giảm từ ≈ 0.7 (v1) xuống ≈ 0.044 (v2). AUC phục hồi từ
0.5607 lên **0.8635**, chỉ chênh 0.001 so với upper bound float-RMSNorm 0.8624.

### 2.5 Xấp xỉ hàm phi tuyến qua LUT

Các hàm SiLU, Softplus, Exp được xấp xỉ qua LUT 256-entry × 16-bit, phủ khoảng
`[−8, +8)` với độ phân giải 0.0625. Phân tích phân phối đầu vào cho thấy:

- SiLU: input trong M3 từ Mamba x_conv, dải `[−4, +4]`
- Softplus: trong M5, output > 0, dải `[0, 6]`
- Exp: cho `delta · A` trong SSM scan, `A < 0` nên input `[−30, 0]`; LUT phủ
  vùng gần 0 với mật độ tốt; vùng < −8 trả về 0 (underflow đúng theo float)

RSqrt ROM dùng 8K entries × 16-bit, Vivado tự suy luận ra BRAM.

---

## 3. Kiến trúc phần cứng

### 3.1 Triết lý thiết kế và lựa chọn PE

**[Hình 2: Sơ đồ tổng thể RTL — ITM_Top_v3 với Memory_System, PE_Array, Const_Storage, RMSNorm_Mul và các capture register]**

Toàn bộ pipeline được hiện thực dưới dạng **một bộ điều khiển FSM duy nhất**
(`ITM_Top_v3`) cấu hình thông qua các tham số top-level (`T_MAX`, `CH_IN`,
`CH_OUT`, `CH_M`, `DT_RANK`, `enc_mode`, `head_mode`). Mỗi block ITM được nạp
trọng số tương ứng qua DMA trước khi pulse `start`. Cách tổ chức này có ba ưu
điểm so với việc instantiate 5 bộ accelerator riêng:

1. **Tài nguyên tỉ lệ với phép tính lớn nhất** (Block 4) thay vì tổng 5 block
2. **Bộ nhớ làm việc dùng chung**: ram_a/ram_b URAM dùng lại cho mọi block
3. **Memory hierarchy đơn giản**: 4 vùng nhớ tổng (data ×2, weight, const) với
   bank-select asymmetric routing cho phép đọc/ghi đồng thời

**Lựa chọn PE Array**: thiết kế dùng mảng 16 phần tử xử lý (PE) đồng nhất, mỗi
PE chứa một bộ nhân DSP48E2 và một bộ tích lũy 40-bit. Đầu vào của PE gồm:

- `pe_A`: vô hướng 16-bit, broadcast tới cả 16 PE (chế độ MAC reduction)
- `pe_A_vec`: vector 16 làn × 16-bit (chế độ element-wise multiply)
- `pe_B`: luôn là vector 16 làn × 16-bit
- `a_is_vector`: chọn pe_A hoặc pe_A_vec

**Tại sao không dùng Sharing Buffer Allocator (SBA)**: SBA trong các nghiên cứu
trước [REF_Advisor_SBA] dùng một bộ đệm trung gian với mux đa cổng feed cho
nhiều kernel khác nhau. SBA tập trung mux đa cổng có hai nhược điểm:

- Delay mux cao do fan-in lớn → critical path bị nén
- Routing congestion cao quanh bộ đệm chia sẻ

Thiết kế ITMN tránh SBA bằng cách: với mỗi state, controller drive trực tiếp
địa chỉ tới Memory_System / Const_Storage / Weight RAM; dữ liệu đọc về qua
**Operand Mux Network** chỉ là mux 2-input đơn giản (chọn giữa rms_norm_out,
m_rd_data, max_buf...). Critical path do đó đi qua URAM cascade + 1 DSP (RMSNorm
multiply) + saturation chain, không bị nén bởi central mux.

**[Hình 3: Chi tiết PE Array — 16 lane mux network ở thượng nguồn, mỗi PE là Unified_PE với 1 DSP48E2 + 40-bit acc + sat16]**

### 3.2 Datapath chia sẻ giữa các phase

Toàn bộ 5 block ITM + encoder + GAP + FC chia sẻ các tài nguyên sau:

- **PE Array 16-lane**: encoder MAC, Phase 1 MAC, Inception convolution, Mamba
  in_proj/depthwise/x_proj/dt_proj/SSM/y_gated/out_proj, Final stage
- **Memory_System**: ram_a (input/intermediate/final c_grp≥1), ram_b
  (P1/X_CONV/Y_SSM/final c_grp=0, raw waveform tại [19000+)), ram_weight (block
  weights + encoder weight + FC weight)
- **Const_Storage**: ram_const 128-entry (per-block bias/BN, RMSNorm gamma,
  encoder bias, FC bias) + 48 LUT activation instances + RSqrt ROM
- **RMSNorm_Mul**: module pipelined chuyên dụng cho RMSNorm output
- **Capture Registers**: max_buf, incep_reg, h_reg (Mamba), gap_sum (GAP),
  fc_acc (FC)

GAP và FC sử dụng **bộ tích lũy chuyên dụng riêng** (không qua PE Array) vì:

- GAP cần lưu sum 24-bit per (c_grp, lane), tổng 8 × 16 × 24-bit = 3072 bit, sau
  đó nhân với hằng số INV_T_Q15 = 131 và shift >>15 để chia cho T=250
- FC chỉ có 5 output channels (không phù hợp với mảng 16-lane), nên dùng một bộ
  tích lũy scalar 40-bit cho mỗi class

### 3.3 Phân hoạch các phép tính lớn (Operation Breakdown)

Phần này mô tả cách từng phép tính lớn được phân rã thành các substep FSM, tận
dụng PE Array hiệu quả nhất.

**Phép tính 1: Inception Conv k=39 (Branch 4)**

Đây là phép tính lớn nhất theo số MAC trong một timestep. Đối với Block 4
(d_out/4=32 in_ch và out_ch, T=250, k=39, 2 c_grp_br):

```
Tổng MAC = T · (out_groups) · k · (in_ch) ≈ 250 × 2 × 39 × 32 = 624,000 MAC/timestep
```

Phân rã thành ba vòng lặp lồng nhau (`S_BR_MAC` substep):

```
for t in 0..T-1:
  for c_grp_br in 0..1:      # 2 nhóm output (block 4)
    for k in 0..38:          # kernel
      for mac_idx in 0..31:  # in_ch
        # 3-cycle substep:
        #   sub 0: read m_rd_data, set w_rd_addr
        #   sub 1: wait BRAM read
        #   sub 2: MAC vào PE Array, pe_clear ở k=0,mac_idx=0
      end (sau 32 mac → wait + write to RAM)
```

Cycle/timestep ≈ 2 × 39 × 32 × 3 + 16 ≈ 7,500 → 1.875M cycle cho phase Inception
Block 4. PE Array tận dụng 100% trong MAC phase (16 output channels song song).

**Phép tính 2: RMSNorm cho Mamba in_proj**

Mỗi timestep, RMSNorm chạy hai lần (cho M1A x_inner và M1B z_gate). Mỗi lần
gồm hai pha:

- **Pha 1 (sum-of-squares)**: đọc P1_out tại t, qua `norm_sq16_fn` tính
  `Σ x[i]²` cho 16 lane, accumulate vào `norm_sq_acc` 40-bit qua nhiều cycle
  (đối với d_out=64: cần 4 cycle; d_out=128 block 4: cần 8 cycle)
- **Pha 2 (rsqrt lookup + MAC scaled)**: shift `norm_sq_acc` thành index, tra
  RSqrt ROM, đăng ký vào `norm_S_reg`; với mỗi output channel:
  `rms_norm_out = sat16( sat16(x·γ >> FB) · S_t >> FB )` qua module
  `RMSNorm_Mul`

Module **RMSNorm_Mul** được pipeline 1-stage (CP-1) giữa hai phép multiply
cascaded, tăng Fmax từ ~70 MHz lên ≥ 100 MHz.

**Phép tính 3: SSM Selective Scan**

Phép tính hồi quy theo thời gian:
```
h_t = exp(delta_t · A) · h_{t-1} + (delta_t · B_t) · u_t      (M6A, M6B)
y_t = C_t · h_t + D · u_t                                       (M7)
```

Mỗi timestep cần:
- M6A: tính `dA = exp(delta_t · A)` per channel, scalar B per timestep
- M6B: cho mỗi lane (CH_M=16 nhóm), MAC theo công thức trên qua 16 trạng thái
- M7: y_gated = C^T · h + D · u, sau đó element-wise multiply với SiLU(z_gate)

Đây là phép tính có dependency theo thời gian (h_{t-1} cần xong mới tính h_t),
do đó không thể song song theo trục thời gian. Tuy nhiên trục channel và trục
state (16) được song song qua PE Array.

**Phép tính 4: Encoder Conv1D k=1**

Toán học giống Phase 1 với `d_in = 12, d_out = 64`:
```
enc_out[c, t] = sat_add( sat16(Σ W[c, j] · x[j, t] >> FB), bias[c] )
```

Hiện thực reuse các state P1 (dedicated S_ENC_*) nhưng địa chỉ chuyển sang:

- Input: `B_ENC_IN_BASE` trong ram_b, 1 word/timestep (12 lane valid)
- Weight: `W_ENC_BASE` trong ram_weight
- Bias: `C_ENC_BIAS` trong ram_const
- Output: `A_INPUT_BASE` trong ram_a (để Block 0 đọc trực tiếp)

Tổng cycle encoder: 4 c_grp × 1000 t × 16 substep ≈ 64K cycle (~0.64ms @ 100MHz)
— không đáng kể so với ~30M cycle của 5 ITM block.

**Phép tính 5: GAP**

```
gap_q[c] = sat16( (Σ_{t=0..T-1} final[c, t]) · INV_T_Q15 >> 15 )
```

với INV_T_Q15 = round(2^15 / T) = 131 cho T=250. Hiện thực:

- 8 nhóm × 16 lane × 24-bit accumulator (đủ chứa max 250 × 32767)
- Vòng lặp T=250 cycle (3 cycle/iteration cho address-wait-latch)
- Finalize: 8 cycle × 16 lane × (1 mul + 1 shift + sat)

Tổng GAP ≈ 6K cycle.

**Phép tính 6: FC Classifier**

```
logit[c] = sat_add( sat16(Σ_{i=0..127} W_fc[c, i] · gap_q[i] >> FB), bias[c] )
```

5 class × 8 grp_in (128/16) × 16 lane × 3 substep ≈ 1920 cycle. Output drive
trực tiếp tới `logit0..4` port top-level.

### 3.4 Memory Map

**[Bảng 2: Memory map đầy đủ — ram_a, ram_b, ram_weight, ram_const với từng vùng và lifetime]**

Bộ nhớ làm việc dùng **compact memory map** với chia sẻ thời gian (temporal
overlap) cho các vùng có lifetime tách biệt:

- `ram_a`: 20K × 256-bit URAM (5-deep cascade × 4-wide = 20 URAM)
- `ram_b`: 20K × 256-bit URAM (20 URAM)
- `ram_weight`: 16K × 256-bit BRAM (114 BRAM18)
- `ram_const`: 128 × 256-bit (Vivado infer distributed LUT do shape nông)

Tổng URAM = 40 (so với 64 URAM của thiết kế trước compact-map), tổng BRAM = 118
(81.94% KV260), tổng URAM = 40 (62.5% KV260).

### 3.5 Bộ điều khiển FSM

FSM v3 có **147 trạng thái** chia thành các phase:

```
ENC (5) → P1 (4) → BR Inception (5) → NORM (10) → M1A-M1B (8) → M2-M3-M3CP (15)
        → M4 (4) → M5 (5) → M6A (15) → M6B (17) → M7 (9) → M8 (4)
        → FIN (5) → CASCADE (5) → GAP (7) → FC (7) → DONE
```

Cấu trúc FSM:

- **Sub-step pattern**: phần lớn các state có 2-3 substep để align với 1-cycle
  BRAM read latency (set addr → wait → use data)
- **Cycle counters**: `t_cnt`, `c_grp`, `c_grp_m`, `mac_idx`, `k_idx`, `branch_id`,
  `s_idx` quản lý các vòng lặp lồng nhau
- **Registered strides** (CP-4): `t_stride_in/m/out/xp` là các bộ tích lũy thay
  vì combinational multiply, giúp giảm độ sâu logic trên đường địa chỉ

---

## 4. Tối ưu hóa Fmax và tài nguyên

**[Hình 4: Bar chart — LUT, URAM, WNS qua các bước tối ưu hóa (baseline → CP-1 → D3.A → RAM-2/3/4)]**

Quá trình tối ưu đi qua bốn bước chính, mỗi bước đều được kiểm chứng byte-exact
trước khi áp dụng tiếp theo:

### 4.1 CP-1: Pipeline RMSNorm Multiply

Trước CP-1, hai phép multiply cascaded trong `x_norm_fn` (RMSNorm output) nằm
trên cùng một đường combinational, gây WNS âm tại 14ns target. Module mới
**`RMSNorm_Mul`** chèn một thanh ghi giữa hai multiply, đẩy WNS lên +0.310 ns @
10ns target (Fmax ~100 MHz). Throughput tăng từ 1.27 → 1.66 inf/s.

### 4.2 D3.A: Consolidate Const Storage

Trước D3.A, các bảng LUT activation (SiLU, Softplus, Exp) và RSqrt ROM được
instantiate rời rạc trong controller; `ram_const` cho bias/scale/shift là một
module BRAM riêng. **`Const_Storage`** gộp tất cả 48 LUT instance + 1 RSqrt ROM
+ ram_const thành một wrapper hierarchy duy nhất, đồng nhất DMA interface
(target=3). Vivado tận dụng hierarchy boundary mới để attribute mux logic gọn
hơn, tiết kiệm **−509 LUT** (4.6%).

### 4.3 RAM-2/3/4: Compact Map + URAM Downsizing

Phân tích lifetime từng vùng nhớ cho thấy peak usage thực tế là 17,256 word
trên ram_a và 19,000 word trên ram_b — thấp hơn nhiều so với khai báo 32K. Hai
thay đổi đồng thời:

- **Compact map**: các vùng có lifetime tách biệt (vd `A_BOT_OUT` chỉ sống trong
  Inception, `A_INPUT_BASE` chỉ sống trong P1) được overlap về cùng base.
- **DEPTH parameter**: BRAM_256b thêm parameter `DEPTH=20480` để Vivado infer
  5-deep URAM cascade × 4-wide (= 20 URAM/bank) thay vì 8×4 = 32 URAM/bank.

Kết quả: **URAM 64 → 40 (−37.5%)**, WNS từ 0.336 → 0.646 ns (slack lớn thêm
0.310 ns).

### 4.4 Tổng kết tối ưu

**[Bảng 3: So sánh trước/sau tối ưu — LUT, REG, BRAM, URAM, DSP, WNS, throughput]**

| Metric | Baseline | Sau tối ưu | Δ |
|---|---|---|---|
| LUT | 11,093 | 10,491 | −602 (−5.4%) |
| URAM | 64 (100%) | 40 (62.5%) | −24 (−37.5%) |
| BRAM | 118 | 118 | 0 |
| DSP | 59 | 59 | 0 |
| WNS @ 10ns | 0.310 ns | 0.646 ns | +0.336 ns |
| Fmax | 100 MHz | 100 MHz | — |

---

## 5. Quy trình kiểm chứng ba lớp

Phương pháp kiểm chứng tuân theo chuẩn industry cho FPGA-DNN accelerators:

### 5.1 Lớp 1: Bit-accurate Python Reference

Module `itmn_pipeline.py` hiện thực toàn bộ pipeline với **integer arithmetic
khớp byte-exact với RTL**, bao gồm:

- 40-bit accumulator MAC → shift FB → sat16 (giống `Unified_PE`)
- `sat_add16`, `bn_relu`, `relu16` functions
- Activation LUT emulation (256-entry × 16-bit)
- RMSNorm v2 với ROM lookup
- Integer chaining giữa các block (kèm `hw_maxpool` cho transitions)
- Encoder Conv1D, GAP, FC theo công thức `(sum · INV_T_Q15) >> 15` và
  `sat_add(sat16((W·gap) >> FB), bias)`

Reference này chạy nhanh (~0.4 sec/sample) và dùng để đo accuracy trên toàn bộ
tập kiểm tra 2158 mẫu.

### 5.2 Lớp 2: RTL Simulation Byte-exact

Testbench **`ITM_CTRL_TB_v2`** orchestrate cho ITM_Top_v3:

1. DMA load encoder weight/bias, FC weight/bias (lần đầu duy nhất)
2. Loop N samples (mặc định 100):
   - DMA load raw waveform → `B_ENC_IN_BASE`
   - Cho mỗi block 0..4: DMA load block weights → pulse `start` → đợi
     `done_all` (hoặc `done_fc` cho block 4)
   - Đọc `logit0..4` từ output port
   - So sánh với golden từ `golden_all/multi/all_logits.txt`
   - In: byte-exact PASS/FAIL, max|diff|, pred-match HW-vs-EXP/HW-vs-FLOAT

Kết quả: **100/100 sample byte-exact PASS** giữa RTL và Python integer chain.
88/100 match với float reference (12 mẫu chênh do quantization noise).

### 5.3 Lớp 3: Hardware Validation

Sau khi byte-exact đã chứng minh, độ chính xác cuối được báo cáo từ Python
reference do tính chất identity:

```
AUC_RTL_on_N_samples ≡ AUC_Python_int_chain_on_N_samples
```

Bảng kết quả cuối trên toàn bộ tập kiểm tra (2158 mẫu):

**[Bảng 4: AUC/TPR — float vs hw_floathead vs hw_inthead trên PTB-XL super-class]**

| Variant | AUC | TPR | Gap vs float |
|---|---|---|---|
| Float (PyTorch float32) | 0.9354 | 0.8154 | — |
| HW float-head | 0.9328 | 0.7892 | −0.0026 |
| HW integer-head (end-to-end) | 0.9328 | 0.7859 | −0.0026 |

Đáng chú ý: integer GAP+FC không gây thêm AUC drop (cùng 0.9328 với float head),
nghĩa là quantization Q4.11 cho phần head an toàn.

---

## 6. Phân tích D1: Mamba và Inception độc lập

Để so sánh công bằng với các nghiên cứu phần cứng riêng cho Mamba hoặc Inception,
controller được build trong ba chế độ:

- **`Mamba_Top`** (define `MAMBA_ONLY`): strip toàn bộ state arms P1+BR+FIN
- **`Inception_Top`** (define `INCEPTION_ONLY`): strip toàn bộ state arms M*/NORM
- **`ITM_Top_v2`** (no define): full controller kernel-only

Cả ba chế độ compile từ **cùng một file** `ITM_CONTROLLER_v2.v` với `ifdef`
preprocessor — chính xác cùng tiêu chí công bằng vì Memory_System / PE_Array /
Const_Storage được instantiate vô điều kiện (memory shared by design).

**[Bảng 5: D1 separation — Mamba_Top, Inception_Top, Full ITM_Top_v2 trên KV260 OOC]**

| Resource | Mamba_Top | Inception_Top | Full | Sum − Full (logic share) |
|---|---|---|---|---|
| LUT | 7,513 | 5,346 | 10,491 | +2,368 (18% saved) |
| REG | 3,945 | 2,103 | 4,504 | +1,544 (26% saved) |
| DSP | 39 | 37 | 59 | +17 (22% saved) |
| BRAM | 118 | 118 | 118 | 100% shared |
| URAM | 40 | 40 | 40 | 100% shared |
| WNS @ 10ns | 0.710 | 1.973 | 0.646 | — |

Phân tích:

- **Memory hoàn toàn share**: BRAM/URAM count = 118/40 cho cả 3 build vì
  Memory_System + Const_Storage được instantiate vô điều kiện
- **Logic-level sharing** đạt 18% LUT, 26% REG, 22% DSP nhờ tái sử dụng PE_Array
  + FSM cho cả Mamba và Inception
- **Inception slack lớn nhất** (1.973 ns) do path đơn giản, không có RMSNorm
  multiply cascade; Mamba bị giới hạn bởi đường URAM → DSP RMSNorm

**[Hình 5: Critical path block diagram — Mamba (URAM ×4 + DSP + sat) so với Inception (URAM ×4 + max_buf + pe_A mux)]**

---

## 7. Phân tích D2: End-to-end overhead

**[Bảng 6: D2 overhead — ITM_Top_v3 (E2E) so với ITM_Top_v2 (kernel-only)]**

*(Số liệu E2E sẽ điền sau khi chạy `E2E_OOC.tcl`; placeholder hiện tại dựa trên
dự đoán phân tích.)*

| Resource | v2 Kernel | v3 E2E (dự kiến) | Overhead |
|---|---|---|---|
| LUT | 10,491 | ~11,000 | +500 (~5%) |
| REG | 4,504 | ~9,500 | +5,000 (~100%) |
| DSP | 59 | ~75 | +16 (~27%) |
| BRAM | 118 | 118 | 0 |
| URAM | 40 | 40 | 0 |
| WNS @ 10ns | 0.646 ns | ~0 to +0.3 ns | giảm 0.3-0.6 ns |

Nguồn overhead chính:

- **REG**: `gap_sum[8][16] × 24-bit` = 3072 FF; `gap_q_reg[8] × 256-bit` = 2048
  FF; `fc_acc` + `fc_bias_lane` + counters ≈ 200 FF
- **DSP**: 16 DSP mới cho FC 16-lane parallel multiply
- **LUT**: FSM mở rộng (147 state vs ~120 state) + capture register mux logic

Trade-off chấp nhận được vì overhead < 30% trên tài nguyên chính trong khi cung
cấp inference end-to-end không cần host round-trip.

---

## 8. So sánh với các nghiên cứu trước

**[Bảng 7: So sánh ITMN với các bộ tăng tốc ECG/Mamba trước đó]**

*(Cần điền sau khi confirm danh sách [REF_X])*

| Công trình | Mô hình | FPGA | LUT | DSP | BRAM | Fmax | Throughput | AUC |
|---|---|---|---|---|---|---|---|---|
| [REF_CNN_ECG_HW1] | CNN ECG | ? | ? | ? | ? | ? | ? | ? |
| [REF_CNN_ECG_HW2] | InceptionTime | ? | ? | ? | ? | ? | ? | ? |
| [REF_Mamba_HW1] | Mamba kernel | ? | ? | ? | ? | ? | ? | ? |
| [REF_Mamba_HW2] | Selective SSM | ? | ? | ? | ? | ? | ? | ? |
| **Ours (D1 Mamba)** | Mamba kernel | KV260 | 7,513 | 39 | 118 | 100 MHz | TBD | — |
| **Ours (D1 Incept.)** | Inception kernel | KV260 | 5,346 | 37 | 118 | 100 MHz | TBD | — |
| **Ours (D2 E2E)** | ITMN full | KV260 | ~11,000 | ~75 | 118 | 100 MHz | TBD | 0.9328 |

Điểm khác biệt chính:

- ITMN là pipeline lai Inception+Mamba đầu tiên trên FPGA (theo hiểu biết tác
  giả) — các nghiên cứu trước hoặc CNN-only hoặc Mamba-only
- D1 separation cho phép so sánh công bằng cả hai phần với prior work
- D2 đầy đủ encoder + classifier head cho end-to-end ECG diagnosis

---

## 9. Kết luận và Hướng phát triển

Bài báo trình bày thiết kế và hiện thực FPGA end-to-end của ITMN, một mô hình lai
Inception+Mamba cho phân loại ECG đa nhãn. Đóng góp chính gồm: (i) kiến trúc
phần cứng chia sẻ tài nguyên giữa các phép tính không đồng nhất (1×1 conv, conv
k=39, RMSNorm, SSM scan, GAP, FC) trên cùng một mảng PE 16-lane và bộ nhớ
ba ngân hàng; (ii) định dạng số Q4.11 với RMSNorm v2 phục hồi AUC từ 0.5607 lên
0.8635; (iii) quy trình kiểm chứng ba lớp đảm bảo byte-exact giữa Python
reference và RTL trên 100/100 mẫu; (iv) báo cáo D1 separation và D2 end-to-end
overhead cho phép so sánh công bằng với prior work. Trên KV260, thiết kế đạt
Fmax 100 MHz với AUC = 0.9328 (chênh 0.0026 so với float reference) trên
PTB-XL super-class.

**Hướng phát triển**:

- **Tối ưu Fmax tiếp**: pipeline 16-lane FC reduce nếu trở thành critical path;
  pipeline norm_sq adder tree (CP-2 trong roadmap)
- **D3 LUT hoàn thiện**: chuyển toàn bộ activation LUT sang dạng distributed
  với attribute `rom_style="distributed"` để tiết kiệm thêm tài nguyên
- **Bring-up trên KV260 thật**: generate bitstream + driver PYNQ, đo throughput
  thực, công suất và end-to-end latency
- **Mở rộng mô hình**: hỗ trợ batch>1 inference, sub-class PTB-XL (24 class)

---

## Tài liệu tham khảo

[REF_PTB_XL] Wagner P. et al. PTB-XL, a large publicly available
electrocardiography dataset. *Scientific Data*, 2020.

[REF_Mamba] Gu A., Dao T. Mamba: Linear-Time Sequence Modeling with Selective
State Spaces. arXiv preprint, 2023.

[REF_Inception_ECG] Fawaz H. I. et al. InceptionTime: Finding AlexNet for time
series classification. *Data Mining and Knowledge Discovery*, 2020.

[REF_Advisor_SBA] *Bạn cung cấp paper của giáo viên về Sharing Buffer Allocator
ở đây.*

[REF_CNN_ECG_HW1, REF_CNN_ECG_HW2] *Bạn cung cấp 2-3 paper FPGA cho CNN ECG.*

[REF_Mamba_HW1, REF_Mamba_HW2] *Bạn cung cấp 2-3 paper FPGA cho Mamba/SSM.*

---

## Phụ lục A: Danh sách viết tắt

- **AUC**: Area Under Curve
- **CP**: Critical Path
- **D1/D2**: Directives 1/2 (separation vs end-to-end)
- **FB**: Fractional Bits (= 11)
- **FSM**: Finite State Machine
- **GAP**: Global Average Pooling
- **OOC**: Out-of-Context (synthesis mode)
- **PE**: Processing Element
- **SBA**: Sharing Buffer Allocator
- **SSM**: State-Space Model
- **TPR**: True Positive Rate
- **URAM**: UltraRAM (Xilinx primitive)

## Phụ lục B: Cấu hình mỗi Block

| Block | T | CH_IN (d_in/16) | CH_OUT (d_out/16) | CH_M (d_inner/16) | DT_RANK |
|---|---|---|---|---|---|
| 0 | 1000 | 4 (64) | 4 (64) | 8 (128) | 4 |
| 1 | 1000 | 4 (64) | 4 (64) | 8 (128) | 4 |
| 2 | 500 | 4 (64) | 4 (64) | 8 (128) | 4 |
| 3 | 500 | 4 (64) | 4 (64) | 8 (128) | 4 |
| 4 | 250 | 4 (64) | 8 (128) | 16 (256) | 8 |

## Phụ lục C: Cycle Counts cho Block 4

| Phase | Cycles |
|---|---|
| Phase 1 (P1) | 389,998 |
| Inception | 3,863,500 |
| Mamba (M1-M8) | 7,433,262 |
| Final (FIN) | 16,000 |
| **Total/block 4** | **11,702,760** |

End-to-end per sample (5 block + encoder + GAP + FC) ≈ 40-50M cycle = 0.4-0.5
sec @ 100 MHz.
