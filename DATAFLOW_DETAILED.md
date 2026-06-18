# Dòng dữ liệu chi tiết: Một mẫu ECG 12 × 1000 đi từ DMA vào đến class output

Tài liệu này theo dõi một mẫu ECG đơn lẻ — vector kích thước (12 lead × 1000
mẫu thời gian) — đi qua toàn bộ pipeline FPGA `ITM_Top_v3`, mô tả từng tín
hiệu, từng state FSM, từng substep, ánh xạ với phép tính PyTorch tương đương.
Mục đích: người đọc nắm được **mọi thao tác phần cứng** thực sự xảy ra để tạo
ra 5 logits ở cuối.

---

## 0. Ký hiệu và quy ước

### 0.1 Tín hiệu top-level chính

| Tín hiệu | Hướng | Width | Ý nghĩa |
|---|---|---|---|
| `clk` | in | 1 | Clock 100 MHz (period 10 ns) |
| `rst` | in | 1 | Async reset, active high |
| `start` | in | 1 | Pulse 1 cycle để khởi động phase hiện tại |
| `T_MAX`, `T_ENC`, `T_GAP` | in | 10 | Số timestep |
| `CH_IN`, `CH_OUT`, `CH_M`, `DT_RANK` | in | 4 | Cấu hình block hiện tại |
| `enc_mode`, `head_mode`, `cascade_mode`, `need_pool` | in | 1 | Chế độ |
| `dma_write_en`, `dma_target[1:0]`, `dma_addr[14:0]`, `dma_wdata[255:0]` | in | — | DMA write từ host |
| `done_encoder`, `done_phase1`, `done_inception`, `done_mamba`, `done_all`, `done_gap`, `done_fc` | out | 1 | Pulse báo từng phase xong |
| `logit0..4` | out | 16 signed | Kết quả phân loại cuối |

### 0.2 Tín hiệu nội bộ chính

| Tín hiệu | Width | Mô tả |
|---|---|---|
| `state` | 8 | Trạng thái FSM hiện tại |
| `substep` | 3 | Sub-step bên trong một state (cho pipeline read-wait-use) |
| `t_cnt` | 10 | Vòng lặp theo trục thời gian |
| `c_grp` | 3 | Vòng lặp theo nhóm output channel (mỗi nhóm = 16 channel) |
| `c_grp_m` | 4 | Vòng lặp theo nhóm Mamba inner channel |
| `mac_idx` | 8 | Index input channel trong MAC reduction |
| `k_idx` | 6 | Index kernel position (cho conv k=9/19/39) |
| `branch_id` | 3 | 0=Bot, 1=B1, 2=B2, 3=B3, 4=B4 |
| `s_idx` | 4 | Index state (0..15) trong SSM scan |
| `c_grp_br` | 1 | Output group con trong nhánh Inception (cho block 4) |
| `m_rd_addr`, `m_wr_addr` | 15 | Địa chỉ đọc/ghi Memory_System (15-bit) |
| `w_rd_addr` | 15 | Địa chỉ đọc Weight RAM |
| `c_rd_addr` | 15 | Địa chỉ đọc Const Storage |
| `m_rd_data`, `m_wr_data` | 256 | Data line, 16 lane × 16-bit |
| `w_rd_data` | 256 | Weight line, 16 lane × 16-bit |
| `c_rd_data` | 256 | Const line, 16 lane × 16-bit |
| `pe_A` | 16 signed | Scalar broadcast vào tất cả 16 PE (chế độ MAC) |
| `pe_A_vec` | 256 | Vector input cho 16 PE (chế độ element-wise) |
| `pe_B` | 256 | Vector input thứ hai cho 16 PE |
| `pe_a_is_vector` | 1 | Chọn `pe_A` (0, broadcast) hay `pe_A_vec` (1, lane-wise) |
| `pe_op_mode` | 2 | MODE_MAC / MODE_MUL / MODE_ADD |
| `pe_clear` | 1 | Reset accumulator trước MAC chuỗi mới |
| `pe_out` | 256 | Output 16 lane × 16-bit từ PE Array |
| `bank_sel` | 1 | Định tuyến read/write giữa ram_a và ram_b |
| `m_we` | 1 | Write enable cho Memory_System |
| `incep_reg` | 256 | Buffer giữ inception output cho FIN |
| `max_buf` | 256 | Accumulator max-pool |
| `h_reg` | 256 | SSM hidden state |
| `norm_sq_acc` | 40 signed | Accumulator sum-of-squares cho RMSNorm |
| `norm_S_reg` | 16 signed | Latched RSqrt ROM output |
| `gap_sum[0:7][0:15]` | 24 signed | Accumulator GAP per (c_grp, lane) |
| `gap_q_reg[0:7]` | 256 | GAP output sau finalize |
| `fc_acc` | 40 signed | Accumulator FC scalar |
| `fc_gap_word` | 256 | Cached gap word cho FC inner loop |

### 0.3 Định dạng Q4.11

- Mỗi giá trị: 16-bit signed two's complement
- 4 bit integer, 11 bit phân số
- Khoảng float biểu diễn: [−16.0, +16.0)
- Độ phân giải: 1/2048 ≈ 0.000488
- Float `f` → int16 `q`: `q = max(-32768, min(32767, floor(f × 2048)))`
- Int16 `q` → float `f`: `f = q / 2048.0`

### 0.4 Một số constants quan trọng

```
FRAC_BITS    = 11
SCALE        = 2048 = 1 << 11
INV_T_Q15    = 131  (= round(2^15 / 250), dùng cho GAP)
N_RMS_PREC   = 6    (RMSNorm v2 extra precision bits)
K_RMS_ROM    = 23170 ≈ sqrt(2^7) × 2048
B_ENC_IN_BASE= 19000  (ram_b)
W_ENC_BASE   = 14000  (ram_weight)
W_FC_BASE    = 14064  (ram_weight)
C_ENC_BIAS   = 64     (ram_const)
C_FC_BIAS    = 68     (ram_const)
A_INPUT_BASE = 0      (ram_a, encoder out / block 0 input)
B_P1_OUT     = 0      (ram_b, P1 output)
```

---

## 1. Khoảnh khắc đầu: vector arrive

### 1.1 Bên ngoài FPGA (host CPU)

Mẫu ECG đến từ tập kiểm tra PTB-XL ở dạng tensor PyTorch `(1, 1000, 12)` —
1 sample × 1000 timestep × 12 lead. Mỗi giá trị là float32 đã chuẩn hóa
(z-score trung bình 0, std 1, phần lớn nằm trong [−5, +5]).

Trong forward path của PyTorch:

```python
def forward(self, x):                    # x: (B, T=1000, C_in=12)
    x = x.transpose(-1, -2)              # → (B, 12, 1000)
    x = self.encoder(x)                  # Conv1d(12→64, k=1) + BN
    x = self.layers(x)                   # 5 ITMBlock + 2 MaxPool
    x = x.mean(dim=-1)                   # GAP → (B, 128)
    x = self.classifier(x)               # Linear(128, 5)
    return x
```

Host chuẩn bị quantize x sang Q4.11 trước khi DMA. Cụ thể `wav_q[ch, t] =
clip(floor(x[t, ch] × 2048), −32768, +32767)` cho mỗi `ch ∈ [0,12)` và mỗi
`t ∈ [0,1000)`. Đây là 12000 giá trị int16.

### 1.2 Pack thành 256-bit word cho DMA

Mỗi 256-bit DMA word chứa 16 lane × 16-bit. Encoder cần input "1 word per
timestep, 12 lane valid + 4 lane zero-pad". Host pack:

```
for t in 0..999:
    word_t = 0
    for ch in 0..11:
        word_t[ch*16 +: 16] = wav_q[ch, t]   # 12 lane đầu
    # lane 12, 13, 14, 15 giữ 0
    DMA_write(target=1, addr=B_ENC_IN_BASE + t, wdata=word_t)
```

Tổng cộng 1000 DMA write transactions sẽ đẩy raw waveform vào `ram_b` ở vùng
[19000, 19999].

### 1.3 Tín hiệu DMA write trong 1 transaction

Ví dụ host muốn ghi waveform của timestep `t=42` vào địa chỉ 19042 trên ram_b:

```
Cycle N (negedge):
    dma_write_en = 1
    dma_target   = 2'b01   // ram_b
    dma_addr     = 15'd19042
    dma_wdata    = {128'h0, wav_q[11,42], wav_q[10,42], ..., wav_q[0,42]}
                   (lane 0 ở LSB của 256-bit)

Cycle N (posedge):
    Memory_System routing logic detect target=1 + write_en:
        we_b      = 1
        addr_a_wr = dma_addr[14:0]   // 14-bit + we_b decide
        din_b     = dma_wdata
    BRAM_256b ram_b internal:
        always @(posedge clk) if (we_a) ram[addr_a] <= din_a;
        → ram_b.ram[19042] <= dma_wdata
    
Cycle N+1 (negedge):
    dma_write_en = 0
    (transaction xong)
```

Quá trình lặp 1000 lần (negedge–posedge–negedge), tổng 2000 clock edge chỉ cho
việc load 1 sample waveform. Ở 100 MHz: 1000 × 10 ns = 10 µs.

### 1.4 Các DMA load khác (boot-time, một lần)

Trước khi xử lý sample, host cũng DMA-load các weights cố định:

- **Encoder weight** (64 × 12 → 64 word ram_weight ở [14000, 14063])
- **Encoder bias** (64 channel → 4 word ram_const ở [64, 67])
- **FC weight** (5 × 128 → 40 word ram_weight ở [14064, 14103])
- **FC bias** (5 channel → 1 word ram_const ở [68])

Mỗi block 0..4 cũng load weight riêng vào ram_weight ở `[0, ~13000)` trước
khi block đó chạy, nhưng encoder/FC weight ở dải `[14000+)` không bị overwrite
vì các block chỉ dùng tới ~13000.

---

## 2. Phase Encoder — chuyển 12 lead → 64 channel

### 2.1 PyTorch tương đương

```python
self.encoder = nn.Sequential(
    nn.Conv1d(12, 64, kernel_size=1),   # weight shape (64, 12, 1)
    nn.BatchNorm1d(64),                 # weight/bias shape (64,)
)
# x: (1, 12, 1000) → conv → (1, 64, 1000) → BN → (1, 64, 1000)
```

Conv1d với k=1 thực chất là phép linear projection per-timestep:
```
out[c, t] = Σ_{j=0..11} W_conv[c, j, 0] · x[j, t] + b_conv[c]
```

Sau khi BN, công thức gộp lại (fuse_conv_bn) thành:
```
W_fused[c, j] = W_conv[c, j, 0] · γ[c] / sqrt(var[c] + eps)
b_fused[c]    = (b_conv[c] - μ[c]) · γ[c] / sqrt(var[c] + eps) + β[c]
out[c, t]     = Σ W_fused[c, j] · x[j, t] + b_fused[c]
```

Trong FPGA, mọi giá trị đã quantize Q4.11. Phép MAC trở thành integer:
```
sum_q[c, t] = Σ_{j=0..11} W_q[c, j] · X_q[j, t]      // 40-bit accumulator
enc_q[c, t] = sat_add16( sat16( sum_q[c, t] >> 11 ), b_q[c] )
```

### 2.2 Memory layout cho Encoder

**Input ram_b[19000..19999]**: 1 word/timestep, lane 0..11 = X[lead 0..11, t],
lane 12..15 = 0.

**Weight ram_weight[14000..14063]**: 64 entries, tổ chức theo `(c_grp_out × 16
+ in_ch)`. Cụ thể tại địa chỉ `W_ENC_BASE + c_grp_out × 16 + in_ch`:
- Lane `i` (i ∈ 0..15) chứa `W_fused[c_grp_out × 16 + i, in_ch]`
- Tức là cùng 1 input channel `in_ch`, 16 output channel khác nhau, packed
  trong 1 word

Bố cục này khớp với cách MAC: với `pe_A` = scalar input channel `in_ch`, và
`pe_B` = 16-lane weight cho 16 output channel, mỗi PE accumulate cho output
channel của lane đó.

**Bias ram_const[64..67]**: 4 word, lane `i` của word `c_grp_out` = `b_fused[
c_grp_out × 16 + i]`.

**Output ram_a[0..3999]**: encoder ghi vào `A_INPUT_BASE + t × 4 + c_grp_out`.
4 word/timestep, mỗi word 16 output channel. Đây cũng là layout mà Block 0
P1 sẽ đọc làm input — encoder output thẳng thay cho input mà host phải DMA.

### 2.3 FSM Encoder — overview

5 state, lặp 4000 lần:

```
S_ENC_MAC (12 substep MAC) → S_ENC_WAIT → S_ENC_WRITE → S_ENC_NEXT
                                                              │
                                                              ├─ next c_grp_out
                                                              └─ hoặc next t
```

Tổng cycle: 4 c_grp × 1000 t × 16 cycle/iteration ≈ 64,000 cycle ≈ 640 µs.

### 2.4 S_IDLE → S_ENC_MAC: kickoff

Host pulse `start = 1` trong khi `enc_mode = 1`, `head_mode = 0`, `T_ENC =
10'd1000`, các config block 0 (`CH_IN=4`, `CH_OUT=4`, ...). Trong cycle này:

```
Cycle 0 (posedge): state = S_IDLE
    if (start) {
        // Clear done flags
        done_phase1 ← 0; done_inception ← 0; done_mamba ← 0
        done_all ← 0; done_encoder ← 0; done_gap ← 0; done_fc ← 0
        // Reset counters
        t_cnt ← 0; c_grp ← 0; c_grp_m ← 0; mac_idx ← 0
        substep ← 0; branch_id ← 0; s_idx ← 0; c_grp_br ← 0
        // Latch mode flags
        enc_mode_reg ← 1; head_mode_reg ← 0
        cascade_mode_reg ← 1; need_pool_reg ← 0
        // Reset GAP/FC state (vì chuẩn bị cho cả pipeline)
        gap_c_grp ← 0; gap_t ← 0; ...
        // Branch: vì enc_mode = 1
        bank_sel ← 1            // sẽ read ram_b, write ram_a
        c_rd_addr ← C_ENC_BIAS  // = 64
        state ← S_ENC_MAC
    }
```

Cuối cycle 0: state đã được lệnh thành S_ENC_MAC, bank_sel = 1, c_rd_addr =
64. Vì c_rd_addr đã thay đổi, Const_Storage sẽ đọc ram_const[64] và đến cuối
cycle 1 `c_rd_data` sẽ valid (BRAM read 1-cycle latency).

### 2.5 Vòng lặp Encoder: chi tiết `(t = 0, c_grp_out = 0)`

Iteration `(t=0, c_grp_out=0)` tính 16 output channel: `enc[0..15, 0]`.

#### Cycle 1: S_ENC_MAC, substep = 0, mac_idx = 0

```
Hardware:
    m_rd_addr ← B_ENC_IN_BASE + t_cnt = 19000 + 0 = 19000
    w_rd_addr ← W_ENC_BASE + c_grp × 16 + mac_idx = 14000 + 0 + 0 = 14000
    pe_A      ← 16'sd0       // chưa dùng, sẽ overwrite ở substep 2
    substep   ← 1

Cuối cycle 1:
    - ram_b internal: addr_b = 19000 đang được đẩy vào port → đầu cycle 2,
      m_rd_data sẽ là word ram_b[19000] = waveform timestep 0
    - ram_weight internal: tương tự, đầu cycle 2 w_rd_data sẽ là encoder
      weight row 0 (in_ch=0)
    - c_rd_data: từ cycle trước đã có ram_const[64] = encoder bias word 0
      (output channel 0..15)
```

#### Cycle 2: S_ENC_MAC, substep = 1, mac_idx = 0

```
Hardware:
    m_rd_data đã valid: = X_q[lane 0..11, t=0]  packed như sau
        m_rd_data[ 0 +: 16] = X_q[0, 0]
        m_rd_data[ 16 +: 16] = X_q[1, 0]
        ...
        m_rd_data[176 +: 16] = X_q[11, 0]
        m_rd_data[192..255]  = 0 (4 lane pad)
    w_rd_data đã valid: 16 lane = W_fused[0..15, in_ch=0]
        w_rd_data[i*16 +: 16] = W_fused_q[i, 0]

    pe_A      ← 16'sd0   // vẫn 0
    substep   ← 2
```

Trong cycle này, FSM chỉ wait 1 cycle để chắc chắn read data đã ổn định
trước khi feed vào PE Array.

#### Cycle 3: S_ENC_MAC, substep = 2, mac_idx = 0

```
Hardware:
    pe_A      ← m_rd_data[mac_idx[3:0] × 16 +: 16] = m_rd_data[0 +: 16] = X_q[0, 0]
    pe_B      ← w_rd_data       // full 256-bit, 16 lane
    pe_clear  ← (mac_idx == 0)  // = 1 → reset accumulator của 16 PE
    // mac_idx != mac_target (= 11), nên next:
    mac_idx   ← 1
    substep   ← 0
```

Tại posedge sắp tới, PE Array sẽ:
- Mỗi PE lane `i`: lấy `pe_a_lane = (a_is_vector ? in_A_vec[i] : in_A) =
  pe_A = X_q[0, 0]` (broadcast)
- Mỗi PE lane `i`: lấy `pe_b_lane = pe_B[i*16 +: 16] = W_fused_q[i, 0]`
- Mỗi PE lane `i`: `mult_i = X_q[0,0] × W_fused_q[i, 0]` (32-bit)
- Vì `pe_clear = 1`: `acc_raw_i ← mult_ext_i` (reset accumulator)
- `out_val_i ← sat16(acc_raw_i >>> 11)` (đăng ký vào FF của PE)

→ 16 PE bắt đầu accumulator chain mới với product đầu tiên.

#### Cycle 4: S_ENC_MAC, substep = 0, mac_idx = 1

```
Hardware:
    m_rd_addr ← 19000 + 0 = 19000    // không đổi, vì t=0 vẫn vậy
    w_rd_addr ← 14000 + 0 + 1 = 14001 // weight row 1, in_ch = 1
    pe_A      ← 0
    substep   ← 1
```

#### Cycle 5: substep = 1, mac_idx = 1

```
m_rd_data đã ổn định ở X_q[0..11, 0] từ trước
w_rd_data mới valid = W_fused_q[0..15, in_ch=1]
substep ← 2
```

#### Cycle 6: substep = 2, mac_idx = 1

```
pe_A      ← m_rd_data[1*16 +: 16] = X_q[1, 0]
pe_B      ← w_rd_data
pe_clear  ← 0    // không reset, tiếp tục accumulate
mac_idx   ← 2
substep   ← 0
```

Tại posedge:
- PE lane `i`: `mult_i = X_q[1, 0] × W_fused_q[i, 1]`
- `acc_raw_i ← acc_raw_i + mult_ext_i`

Sau 12 lần lặp như vậy (mac_idx = 0..11), accumulator của PE lane `i` chứa
`Σ_{j=0..11} X_q[j, 0] × W_fused_q[i, j]` = sum theo công thức encoder.

#### Cycle 36: S_ENC_MAC, substep = 2, mac_idx = 11 (lần cuối)

```
pe_A     ← X_q[11, 0]
pe_B     ← w_rd_data
pe_clear ← 0
// mac_idx == 11 (= max), nên:
state ← S_ENC_WAIT
mac_idx & substep giữ nguyên cho cycle này
```

#### Cycle 37: S_ENC_WAIT

```
state ← S_ENC_WRITE
```

Wait 1 cycle để PE Array commit kết quả vào `pe_out` (Unified_PE có
registered output).

#### Cycle 38: S_ENC_WRITE

```
m_we      ← 1
bank_sel  ← 1   // route write to ram_a
m_wr_addr ← A_INPUT_BASE + t_cnt × 4 + c_grp = 0 + 0 + 0 = 0
m_wr_data: each lane i ←
    sat_add16(pe_out[i*16 +: 16], c_rd_data[i*16 +: 16])
    // pe_out[i*16 +: 16] = sat16(accumulator_i >>> 11) ≈ MAC result for output channel i
    // c_rd_data[i*16 +: 16] = b_fused_q[i] (bias for output channel i)
state ← S_ENC_NEXT
```

Tại posedge:
- ram_a.we_a = 1, addr_a = 0, din_a = m_wr_data → ram_a[0] ← m_wr_data
- m_wr_data lane `i` = `enc_q[i, 0]` (encoder output cho output channel `i`,
  timestep 0)

→ Sau cycle 38, ram_a[0] chứa encoder output 16 channel đầu (output ch 0..15)
cho timestep 0.

#### Cycle 39: S_ENC_NEXT

```
mac_idx ← 0
substep ← 0
m_we    ← 0  // (cũng đã set 0 mặc định ở đầu always block)

// c_grp (= 0) != 3, nên:
c_grp     ← 1
c_rd_addr ← C_ENC_BIAS + c_grp + 1 = 64 + 0 + 1 = 65  // next bias word
state     ← S_ENC_MAC
```

Lưu ý: `c_rd_addr` được set sớm để có thời gian Const_Storage đọc xong word
mới trước khi cần ở cycle S_ENC_WRITE tiếp theo.

### 2.6 Lặp lại cho c_grp_out = 1..3 (t = 0)

Tương tự như c_grp_out = 0:

- `(t=0, c_grp_out=1)`: tính output channel 16..31, ghi vào ram_a[1]
- `(t=0, c_grp_out=2)`: tính output channel 32..47, ghi vào ram_a[2]
- `(t=0, c_grp_out=3)`: tính output channel 48..63, ghi vào ram_a[3]

Cycle count cho một (t, c_grp): 
- substep 0/1/2 cho mac_idx = 0..11: 12 × 3 = 36 cycle
- S_ENC_WAIT: 1 cycle
- S_ENC_WRITE: 1 cycle
- S_ENC_NEXT: 1 cycle (mặc dù không strictly đợi, FSM vẫn tiêu cycle này)

Total: 39 cycle/iteration ≈ 16 (làm tròn nhưng thực ra 39 với detailed)

### 2.7 Chuyển timestep: t = 0 → t = 1

Khi đã hoàn thành `(t=0, c_grp=3)`, tại cycle S_ENC_NEXT:

```
mac_idx ← 0
substep ← 0
m_we    ← 0

// c_grp == 3, nên:
c_grp ← 0
// t_cnt != T_ENC - 1, nên:
t_cnt     ← 1   // hoặc t_cnt + 1
c_rd_addr ← C_ENC_BIAS  // = 64, reset cho c_grp = 0
state     ← S_ENC_MAC
```

Iteration `(t=1, c_grp_out=0)` lặp y hệt nhưng `m_rd_addr = 19001` (đọc
waveform timestep 1), `m_wr_addr = A_INPUT_BASE + 1 × 4 + 0 = 4`.

### 2.8 Kết thúc Encoder: t = 999, c_grp = 3

Sau khi ghi xong word cuối cùng tại ram_a[3999] (= A_INPUT_BASE + 999×4 + 3),
FSM rơi vào S_ENC_NEXT lần cuối:

```
mac_idx ← 0; substep ← 0; m_we ← 0
// c_grp == 3 (cuối)
c_grp ← 0
// t_cnt == T_ENC - 1 == 999, nên: encoder done
done_encoder     ← 1   // pulse 1 cycle
enc_phase        ← 0
t_cnt            ← 0
t_cnt_zero       // reset registered strides t_stride_in/m/out/xp về 0
bank_sel         ← 0   // back to P1 mode: read ram_a, write ram_b
c_rd_addr        ← C_P1_BIAS  // = 0
need_pool_reg    ← need_pool  (re-latch, vẫn 0 cho block 0)
cascade_mode_reg ← cascade_mode  (vẫn 1)
t_out_cnt        ← 0
state            ← S_P1_MAC
```

→ Bắt đầu Block 0 P1 ngay lập tức, KHÔNG cần host pulse start lại.

Sau encoder:
- `ram_a[0..3999]` chứa encoder output: `enc_q[0..63, 0..999]` packed
  16 channel/word, 4 word/timestep, 1000 timestep
- Tất cả memory khác chưa được ghi (block 0 weight đã có sẵn từ DMA boot)

---

## 3. Phase 1 — Block 0 P1 (Conv1D 1×1 + BN, 64 → 64)

### 3.1 PyTorch tương đương

```python
class ITMBlock:
    def __init__(...):
        self.conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
        )
# P1 = self.conv(x), với x là encoder output (1, 64, 1000)
```

Sau fuse_conv_bn, công thức P1:
```
p1[c, t] = Σ_{j=0..63} W_p1[c, j] × enc[j, t] + b_p1[c]
```

Kích thước MAC: 64 input channel × 64 output channel, T = 1000 timestep.

### 3.2 Memory layout

**Input ram_a[0..3999]**: từ encoder. 4 word/timestep, mỗi word 16 channel.
- ram_a[t × 4 + c_grp_in] với c_grp_in ∈ [0, 3]
- Lane `i` của word này = `enc_q[c_grp_in × 16 + i, t]`

**Weight ram_weight[0..255]**: P1 cho block 0.
- W_P1_BASE = 0
- Tại địa chỉ `W_P1_BASE + c_grp_out × d_in + mac_idx` (d_in = 64, c_grp_out
  ∈ [0,4), mac_idx ∈ [0, 64))
- Lane `i` chứa `W_p1_q[c_grp_out × 16 + i, mac_idx]`

**Bias ram_const[0..3]**: 4 word, lane `i` của word `c_grp_out` =
`b_p1_q[c_grp_out × 16 + i]`.

**Output ram_b[B_P1_OUT + t × 4 + c_grp_out]**: 4 word/timestep, mỗi word 16
output channel.

### 3.3 FSM P1 — overview

```
S_P1_MAC (3 substep, mac_idx 0..63) → S_P1_WAIT → S_P1_WRITE → S_P1_NEXT
```

Cycle count: 4 c_grp × 1000 t × (3 × 64 + 3) = 4 × 1000 × 195 ≈ 780,000 cycle.

### 3.4 Vòng lặp P1: detailed (t = 0, c_grp_out = 0)

#### Cycle 0 (kế thừa từ S_ENC_NEXT): S_P1_MAC, substep = 0, mac_idx = 0

```
m_rd_addr ← A_INPUT_BASE + t_stride_in + mac_idx[7:4]
            = 0 + 0 + 0 = 0   // word đầu tiên của t=0
            // mac_idx[7:4] = 0 vì mac_idx = 0
w_rd_addr ← W_P1_BASE + c_grp × d_in + mac_idx
            = 0 + 0 × 64 + 0 = 0
pe_A      ← 0
substep   ← 1
```

Lưu ý: P1 dùng `t_stride_in` (registered accumulator). Tại đầu phase này
`t_stride_in = 0` (đã reset bởi `t_cnt_zero` khi vào). Stride increment mỗi
khi t tăng: `t_stride_in += CH_IN = 4`.

#### Cycle 1: substep = 1, mac_idx = 0

```
m_rd_data valid: ram_a[0] = encoder out cho (ch 0..15, t=0)
w_rd_data valid: ram_weight[0] = W_p1_q[0..15, 0]   // 16 output ch, input ch 0
pe_A      ← 0
substep   ← 2
```

#### Cycle 2: substep = 2, mac_idx = 0

```
pe_A      ← m_rd_data[mac_idx[3:0] × 16 +: 16] = m_rd_data[0 +: 16] = enc_q[0, 0]
pe_B      ← w_rd_data
pe_clear  ← (mac_idx == 0) = 1   // reset accumulator
mac_idx   ← 1
substep   ← 0
```

PE Array tại posedge:
- 16 PE: `mult_i = enc_q[0, 0] × W_p1_q[i, 0]`
- Reset accumulator: `acc_raw_i ← mult_ext_i`

#### Cycle 3: substep = 0, mac_idx = 1

```
m_rd_addr ← A_INPUT_BASE + t_stride_in + mac_idx[7:4]
            = 0 + 0 + 0 = 0   // mac_idx = 1, [7:4] = 0 → cùng word
w_rd_addr ← 0 + 0 + 1 = 1
substep   ← 1
```

#### Cycle 4: substep = 1, mac_idx = 1

```
w_rd_data valid: ram_weight[1] = W_p1_q[0..15, 1]
m_rd_data: vẫn ram_a[0]
substep ← 2
```

#### Cycle 5: substep = 2, mac_idx = 1

```
pe_A     ← m_rd_data[1 × 16 +: 16] = enc_q[1, 0]
pe_B     ← w_rd_data
pe_clear ← 0
mac_idx  ← 2
```

... (lặp 15 lần với mac_idx[7:4] = 0, cùng word ram_a[0])

#### Cycle 47: substep = 2, mac_idx = 15

```
pe_A     ← enc_q[15, 0]
pe_B     ← w_rd_data = W_p1_q[0..15, 15]
pe_clear ← 0
mac_idx  ← 16
```

#### Cycle 48: substep = 0, mac_idx = 16

```
m_rd_addr ← A_INPUT_BASE + t_stride_in + mac_idx[7:4]
            = 0 + 0 + 1 = 1   // mac_idx = 16, [7:4] = 1 → word kế ram_a[1]
w_rd_addr ← 0 + 0 + 16 = 16
substep   ← 1
```

#### Cycle 49: substep = 1, mac_idx = 16

```
m_rd_data NEW valid: ram_a[1] = encoder out cho (ch 16..31, t=0)
w_rd_data NEW valid: W_p1_q[0..15, 16]
substep ← 2
```

#### Cycle 50: substep = 2, mac_idx = 16

```
pe_A     ← m_rd_data[mac_idx[3:0] × 16 +: 16] = m_rd_data[0 +: 16] = enc_q[16, 0]
pe_B     ← w_rd_data
pe_clear ← 0
mac_idx  ← 17
```

... (tiếp tục đến mac_idx = 63)

#### Cycle 191: substep = 2, mac_idx = 63 (lần cuối)

```
pe_A     ← m_rd_data[15 × 16 +: 16] = enc_q[63, 0]
pe_B     ← w_rd_data = W_p1_q[0..15, 63]
pe_clear ← 0
// mac_idx == d_in_last == 63
state ← S_P1_WAIT
```

#### Cycle 192: S_P1_WAIT

```
state ← S_P1_WRITE
```

#### Cycle 193: S_P1_WRITE

```
m_we      ← 1
m_wr_addr ← B_P1_OUT + t_stride_out + c_grp = 0 + 0 + 0 = 0
m_wr_data: each lane i ←
    sat_add16(pe_out[i*16 +: 16], c_rd_data[i*16 +: 16])
state ← S_P1_NEXT
```

bank_sel hiện = 0, nên write routes to ram_b. ram_b[0] ← P1 output cho
(ch 0..15, t=0).

#### Cycle 194: S_P1_NEXT

```
mac_idx ← 0
substep ← 0
// c_grp == 0, ch_out_last[2:0] == 3, nên:
c_grp     ← 1
c_rd_addr ← C_P1_BIAS + c_grp + 1 = 0 + 0 + 1 = 1   // next bias word
state     ← S_P1_MAC
```

### 3.5 Tổng kết cycle cho 1 (t, c_grp) iteration

| Phase | Cycle |
|---|---|
| 64 × MAC substeps (3 cycle each) | 192 |
| S_P1_WAIT | 1 |
| S_P1_WRITE | 1 |
| S_P1_NEXT | 1 |
| **Total** | **195** |

Per block 0 P1: 4 c_grp × 1000 t × 195 = 780,000 cycle.

### 3.6 Chuyển timestep: kích hoạt t_cnt_inc

Khi c_grp đã hoàn thành 0..3 cho timestep t hiện tại, S_P1_NEXT:

```
mac_idx ← 0; substep ← 0
// c_grp == 3 (cuối)
c_grp ← 0
// t_cnt != t_last (= 999)
t_cnt      ← t_cnt + 1
t_cnt_inc;  // task: increment all 4 registered strides
   t_stride_in  ← t_stride_in + CH_IN  // += 4
   t_stride_m   ← t_stride_m + CH_M     // += 8
   t_stride_out ← t_stride_out + CH_OUT // += 4
   t_stride_xp  ← t_stride_xp + 3       // += 3 (cho M4)
c_rd_addr ← C_P1_BIAS  // reset cho c_grp=0
state     ← S_P1_MAC
```

→ Stride register update đồng bộ với t_cnt, không cần combinational
multiplier trên data path (đây là CP-4 fix tăng Fmax).

### 3.7 Kết thúc P1: chuyển sang Inception

Sau khi P1 xong với t = 999, c_grp = 3, S_P1_NEXT:

```
mac_idx ← 0; substep ← 0
// c_grp == 3 (cuối)
c_grp ← 0
// t_cnt == t_last == 999
done_phase1 ← 1   // pulse
t_cnt        ← 0; t_cnt_zero  // reset strides
k_idx        ← 0
branch_id    ← 0   // start với nhánh Bot
bank_sel     ← 1   // (Inception BR read ram_b P1_out, write ram_a BOT_OUT)
c_grp_br     ← 0   // start với group output đầu (cho block 4)
state        ← S_BR_MAC
```

Sau Phase 1:
- `ram_b[0..3999]` chứa P1_out: `p1_q[c, t]` cho c ∈ [0, 63], t ∈ [0, 999]
- `ram_a[0..3999]` vẫn chứa encoder out (sẽ bị overwrite ở các phase sau khi
  cần)
- done_phase1 đã = 1

---

## 4. Phase 2 — Inception (Block 0)

### 4.1 PyTorch tương đương

```python
class BaseInceptionBlock:
    def __init__(self, d_model=64):
        dim = d_model // 4   # = 16 cho block 0
        self.bottleneck = nn.Conv1d(d_model, dim, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(dim, dim, kernel_size=39, padding=19, bias=False)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=19, padding=9, bias=False)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=9, padding=4, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(d_model, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        bot = self.bottleneck(x)            # (1, 16, 1000)
        out4 = self.conv4(bot)              # (1, 16, 1000)
        out3 = self.conv3(bot)              # (1, 16, 1000)
        out2 = self.conv2(bot)              # (1, 16, 1000)
        out1 = self.conv1(self.maxpool(x))  # (1, 16, 1000) (maxpool input, conv1 k=1)
        cat  = torch.cat((out1, out2, out3, out4), dim=1)  # (1, 64, 1000)
        return self.relu(self.bn(cat))
```

Tổng cộng 5 nhánh tính song song trong PyTorch, gộp lại thành 64 channel.
**Trên FPGA, 5 nhánh tính tuần tự** (chia sẻ PE Array).

### 4.2 Thứ tự nhánh + memory layout

FSM lặp `branch_id = 0..4`:

| branch_id | Nhánh | PyTorch op | Input | Weight | Output | Kernel |
|---|---|---|---|---|---|---|
| 0 | Bot | `bottleneck(p1)` | B_P1_OUT (ram_b) | W_BOT_BASE | A_BOT_OUT (ram_a) | k=1 |
| 1 | B1 | `conv1(maxpool(p1))` | B_P1_OUT | W_B1_BASE | A_CH1_OUT | k=1 + maxpool |
| 2 | B2 | `conv2(bot)` | A_BOT_OUT | W_B2_BASE | B_CH2_OUT | k=9, pad=4 |
| 3 | B3 | `conv3(bot)` | A_BOT_OUT | W_B3_BASE | B_CH3_OUT | k=19, pad=9 |
| 4 | B4 | `conv4(bot)` | A_BOT_OUT | W_B4_BASE | B_CH4_OUT | k=39, pad=19 |

Mỗi nhánh sản xuất `dim = 16` channel output. Sau khi cả 5 nhánh xong, FIN
phase sẽ concat + BN + ReLU.

### 4.3 Nhánh Bot (branch_id = 0): chi tiết

Bot là 1×1 conv giống encoder/P1 nhưng giảm channel 64 → 16. Khác P1 ở:
- Input từ B_P1_OUT (bank B), output vào A_BOT_OUT (bank A)
- Không bias, không BN (per BaseInceptionBlock spec)
- Chỉ 1 c_grp_br loop (vì dim=16 = 1 group cho block 0)

#### State arms trong S_BR_MAC (chỉ phần liên quan Bot)

`is_ch64_branch` = (branch_id == 0 || branch_id == 1) = true cho Bot/B1.
`current_num_in_ch` = d_out = 64 cho Bot.
`current_kernel` = 1.
`current_pad` = 0.
`is_padding` = false (k=1, pad=0).
`mac_target` = 63.

#### Cycle (t=0, c_grp_br=0, k_idx=0, mac_idx=0): S_BR_MAC substep = 0

```
m_rd_addr ← current_data_base + (t_eff × CH_OUT) + mac_idx[7:4]
          = B_P1_OUT + 0 + 0 = 0
w_rd_addr ← current_w_base + br_w_offset + (k_idx × current_num_in_ch) + mac_idx
          = W_BOT_BASE + 0 + 0 + 0
pe_A     ← 0
substep  ← 1
```

#### Cycle next: substep = 1

```
m_rd_data valid: ram_b[0] = p1_q[0..15, 0]   // bank_sel = 1 reads ram_b
w_rd_data valid: ram_weight[W_BOT_BASE] = W_bot_q[0..15, 0]
substep ← 2
```

#### Cycle next: substep = 2

```
pe_A     ← m_rd_data[0 +: 16] = p1_q[0, 0]
pe_B     ← w_rd_data
pe_clear ← (k_idx == 0 && mac_idx == 0) = 1
mac_idx  ← 1
substep  ← 0
```

PE Array: `mult_i = p1_q[0, 0] × W_bot_q[i, 0]`, accumulator reset.

#### Repeat 63 lần (mac_idx = 1..63), tương tự P1 logic

Sau khi MAC qua tất cả 64 input channel:

```
state ← S_BR_WAIT (substep = 2, mac_idx = 63, k_idx = 0)
```

#### S_BR_WAIT → S_BR_WRITE

```
m_we      ← 1
m_wr_addr ← current_out_base + (t_cnt × br_dim_groups) + c_grp_br
          = A_BOT_OUT + 0 + 0 = 0
m_wr_data ← pe_out   // 16 lane = bottleneck output cho 16 channel
// bank_sel = 1, write to ram_a (A_BOT_OUT)
state ← S_BR_NEXT
```

Lưu ý: Bot không bias, nên `m_wr_data = pe_out` thẳng, không `sat_add16`.

#### S_BR_NEXT

```
m_we ← 0; mac_idx ← 0; k_idx ← 0; substep ← 0
// k_idx (= 0) == k_target (= 0) cho Bot → kernel xong
// c_grp_br (= 0) == br_grp_last (= 0) cho block 0 → c_grp_br xong
// t_cnt != t_last → next timestep
t_cnt     ← t_cnt + 1; t_cnt_inc
state     ← S_BR_MAC
```

Total cho 1 (t, c_grp_br) iteration của Bot: ~64 × 3 + 3 = 195 cycle. Tổng
Bot: 1000 × 195 ≈ 195,000 cycle.

### 4.4 Nhánh B1: MaxPool input + Conv 1×1

B1 đọc P1_OUT[t-1], P1_OUT[t], P1_OUT[t+1], lấy elementwise max, rồi MAC qua
W_B1.

#### S_BR_MAC khi branch_id = 1: substep pattern phức tạp hơn

```
substep 0: read p1_q[t-1]              (set m_rd_addr cho b1_t_prev)
substep 1: wait
substep 2: latch p1_q[t-1] → max_buf,
           set m_rd_addr cho t_cnt (b1_t)
substep 3: wait
substep 4: latch elem_max16(max_buf, p1_q[t]) → max_buf,
           set m_rd_addr cho b1_t_next
substep 5: wait
substep 6: pe_A ← max_buf[mac_idx[3:0] × 16 +: 16],
           tính elem_max final với m_rd_data,
           pe_B ← w_rd_data, MAC
```

Edge case `t = 0`: b1_t_prev = 0 (giới hạn dưới). `t = 999`: b1_t_next = 999
(giới hạn trên).

Cycle/iteration: phức tạp hơn Bot, nhưng cùng cấu trúc lặp mac_idx = 0..63.

### 4.5 Nhánh B2/B3/B4: Conv k=9/19/39 trên Bot output

B2 (k=9), B3 (k=19), B4 (k=39) đều conv trên A_BOT_OUT (16 input channel,
chứ không phải 64). Vì vậy:

- `current_num_in_ch = dim = 16` cho block 0
- `mac_idx` chỉ chạy 0..15 (1 word/timestep)
- Phải lặp qua kernel position `k_idx = 0..k-1`
- Padding handling: nếu `t_eff = t + k_idx - pad < 0` hoặc `>= T`, set
  `pe_A = 0` (zero padding)

#### Vòng lặp B4 (k=39, branch_id=4)

Cấu trúc 4 vòng lồng nhau:
```
for c_grp_br in 0..br_grp_last:   # 0 cho block 0 (chỉ 1 group)
  for t in 0..T-1:
    for k_idx in 0..38:
      for mac_idx in 0..15:
        # 3 substep MAC
```

Per timestep: 39 × 16 × 3 + 3 ≈ 1875 cycle. Total B4 block 0: 1000 × 1875 =
1.875M cycle.

Đây là phép tính **dài nhất trong Inception**. Lý do: B4 có kernel 39 và phải
quét toàn bộ T=1000 timestep.

### 4.6 Tổng kết Phase 2

Cycle count cho Inception block 0:
- Bot: 195K
- B1: ~250K (do extra substeps maxpool)
- B2: 9 × 16 × 3 × 1000 = 432K
- B3: 19 × 16 × 3 × 1000 = 912K
- B4: 39 × 16 × 3 × 1000 = 1.872M
- **Total**: ~3.66M cycle

Memory sau Phase 2:
- A_BOT_OUT (ram_a[0..999]): 16 channel × 1000 t = 1000 word (overlap với
  A_INPUT_BASE, nhưng input không còn cần)
- A_CH1_OUT (ram_a[16000..16999]): B1 output, 1000 word
- B_CH2_OUT (ram_b[16000..16999]): B2 output, 1000 word
- B_CH3_OUT (ram_b[17000..17999]): B3 output
- B_CH4_OUT (ram_b[18000..18999]): B4 output

Sau cùng, S_BR_NEXT cuối:
```
done_inception ← 1   // pulse
t_cnt ← 0; t_cnt_zero
bank_sel ← 1   // chuẩn bị cho NORM (đọc B_P1_OUT)
norm_sq_acc ← 40'd0   // reset RMSNorm accumulator
state ← S_NORM_M1A_SQ_READ
```

→ Chuyển sang Mamba phase, bắt đầu bằng RMSNorm.

---

## 5. Phase 3 — Mamba (Block 0)

Đây là phase phức tạp nhất, gồm các sub-phase:

1. **NORM_M1A**: RMSNorm cho M1A
2. **M1A**: in_proj_x (linear 64 → 128 = d_inner) qua RMSNorm output
3. **NORM_M1B**: RMSNorm lần 2 cho M1B
4. **M1B**: in_proj_z (linear 64 → 128)
5. **M2**: depthwise conv1d k=4 + bias trên x_inner
6. **M3**: SiLU(x_conv) → u_safe
7. **M3CP**: copy u_safe sang B_U_SAFE (do bank constraints)
8. **M4**: x_proj (linear d_inner → dt_rank + 2 × d_state) → 3 output groups
9. **M5**: dt_proj + bias + softplus → delta_t
10. **M6A**: tính dA = exp(delta_t × A) per channel + scalar B
11. **M6B**: SSM scan h_t = dA × h_{t-1} + dB × u → y per state
12. **M7**: y_gated = y × SiLU(z_gate), tính ở cùng state
13. **M8**: out_proj (linear d_inner → d_in = 64)

### 5.1 NORM_M1A: RMSNorm per timestep

#### PyTorch

```python
class RMSNorm:
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * self.weight
```

Cho x: (B, T, d) = (1, T, 64) đối với block 0. Trục `-1` là channel dim.
Vì input vào Mamba block là `x.transpose` để chiều cuối thành d, vì vậy
mỗi timestep tính một scalar `S_t = rsqrt(mean(x²_t))`, sau đó nhân
`x[:, t, :] × S_t × γ`.

#### FPGA — 2 phase

**Phase 1: sum-of-squares**

```
for t in 0..T-1:
  norm_sq_acc ← 0
  for c_grp_in in 0..3:  # 4 group × 16 lane = 64 channel
    read p1_q[c_grp_in × 16..+15, t] from ram_b[t × 4 + c_grp_in]
    norm_sq_acc += Σ_{i=0..15} (lane_i)²    # norm_sq16_fn
  # tại đây norm_sq_acc = Σ_{c=0..63} p1_q[c, t]²

  mean_i = norm_sq_acc >> (log2(d) + 2*FB - 1 - N)
         = norm_sq_acc >> (6 + 22 - 1 - 6) = >> 21    # cho d=64
  norm_S_reg ← RSqrt_ROM[clip(mean_i, 0, 8191)]
```

**Phase 2: x * γ * S_t** (đã tính `norm_S_reg`):

```
for c_grp_out in 0..7 (= CH_M = 8 cho block 0): # M1A có d_out = d_inner = 128
  for mac_idx in 0..63:                          # d_in = 64
    read p1_q[mac_idx, t] và γ[mac_idx]
    # x_norm = sat16( sat16(p1 × γ >> 11) × norm_S_reg >> 11 )
    # rms_norm_out (1 scalar)
    pe_A ← rms_norm_out
    pe_B ← W_M1A_X[c_grp_out × 16..+15, mac_idx]
    MAC vào PE Array (16 output channel song song)
  write x_inner[c_grp_out × 16..+15, t] to A_X_INNER[t × 8 + c_grp_out]
```

#### State chi tiết NORM_M1A_SQ_READ → S_NORM_M1A_SQ_LATCH

```
S_NORM_M1A_SQ_READ:    set m_rd_addr ← B_P1_OUT + t_stride_in + mac_idx[7:4]
S_NORM_M1A_SQ_WAIT:    wait
S_NORM_M1A_SQ_LATCH:   norm_sq_acc ← norm_sq_acc + norm_sq16_fn(m_rd_data)
                       (combinational reduce 16 squares + sum)
S_NORM_M1A_SQ_NEXT:    advance mac_idx; nếu xong 4 c_grp_in cho t này → S_NORM_M1A_MEAN
S_NORM_M1A_MEAN:       compute mean_i, look up RSqrt_ROM, latch norm_S_reg
                       → S_M1A_MAC
```

`norm_sq16_fn` là combinational function: nhận 256-bit (16 lane × 16-bit),
tính 16 × 16-bit² = 16 × 32-bit, cộng cây tree 16-way thành 40-bit.

Sau khi `norm_sq_acc` xong cho timestep `t`, chuyển vào pha 2 (M1A_MAC).

### 5.2 M1A: x_inner projection (linear 64 → 128)

#### Cycle (t = 0, c_grp_out = 0, mac_idx = 0): S_M1A_MAC substep = 0

```
m_rd_addr ← B_P1_OUT + t_stride_in + mac_idx[7:4]
          = 0 + 0 + 0 = 0
w_rd_addr ← W_M_X_BASE + c_grp_out × d_out + mac_idx
          = W_M_X_BASE + 0 + 0
c_rd_addr ← C_NORM_W + (mac_idx[7:4])
          = 56 + 0 = 56 (gamma[0..15] cho input channel 0..15)
substep   ← 1
```

#### substep = 1: wait

#### substep = 2: compute rms_norm_out + MAC

```
// rms_norm_out tính qua RMSNorm_Mul module:
//   p1_wide = p1 × γ          (1 DSP)
//   p1_reg  = sat16(p1_wide >>> 11)  (registered, pipelined)
//   out_wide = p1_reg × norm_S_reg (1 DSP)
//   rms_norm_out = sat16(out_wide >>> 11)
pe_A     ← rms_norm_out
pe_B     ← w_rd_data
pe_clear ← (mac_idx == 0)
mac_idx  ← 1
substep  ← 0
```

Sau 64 MAC (mac_idx = 0..63), accumulator của PE lane `i` = `Σ_j rms_norm
_out[j] × W_M1A_x_q[c_grp_out × 16 + i, j]`.

#### S_M1A_WRITE

```
m_we      ← 1
m_wr_addr ← A_X_INNER + t_stride_out + c_grp_out  // (t × CH_M + c_grp_out)
// CH_M = 8 cho block 0 → t_stride_out tăng +CH_OUT mỗi t; nhưng cho M1A
// dùng t_stride_m thay (tăng +CH_M)
m_wr_data ← pe_out  // không bias cho M1A
state ← S_M1A_NEXT
```

Lưu ý: A_X_INNER overlap với A_INPUT_BASE (cùng base 0), nhưng A_INPUT_BASE
giờ đã DEAD (P1 đã đọc xong, encoder out đã consume). Compact memory map
cho phép overlap này.

#### M1A NEXT iteration

`c_grp_out` từ 0..7 cho block 0 (CH_M = 8 → 128 / 16 = 8 group). Lặp 8 lần cho
mỗi timestep, sau đó NORM lặp lại cho timestep tiếp theo.

**Tổng cycle M1A 1 timestep**: NORM (~70) + M1A (8 × 195 = 1560) ≈ 1630 cycle.

### 5.3 M1B: z_gate projection (giống M1A)

Cùng cấu trúc với M1A nhưng:
- Weight ở W_M_Z_BASE (z_gate weights)
- Output ghi vào A_Z_GATE (ram_a, [8000..15999])
- Cần RMSNorm lần 2 (vì PE Array đã ghi đè state)

Tổng cycle M1B 1 timestep ≈ 1630 cycle.

**Cycle cho M1A + M1B per timestep**: ~3260 cycle.

### 5.4 M2: Depthwise Conv1D k=4 + bias

#### PyTorch

```python
self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, groups=d_inner,
                       padding=3, bias=True)
# x_conv = conv1d(x_inner)
```

Depthwise = mỗi output channel chỉ phụ thuộc 1 input channel cùng index.
Pad = 3 (causal pad bên trái).

```
x_conv[c, t] = Σ_{k=0..3} W_dw[c, k] × x_inner[c, t-3+k] + b_dw[c]
              (với padding 0 khi t-3+k < 0)
```

#### FPGA — Depthwise = element-wise multiply

Vì mỗi channel độc lập (groups=d_inner), không có MAC reduction qua channels.
PE Array dùng chế độ `a_is_vector=1, op_mode=MUL`.

#### Cycle (t, c_grp_m, k_idx=0): S_M2_MAC

```
m_rd_addr ← A_X_INNER + m2_t_stride + c_grp_m
            // m2_t_eff = t_cnt + k_idx - 3
w_rd_addr ← W_M_DW_BASE + c_grp_m × 4 + k_idx
pe_A_vec  ← m_rd_data    // 16 lane × 16-bit, mỗi lane = x_inner[c_grp_m×16+i, t-3+k_idx]
pe_B      ← w_rd_data     // 16 lane × 16-bit, lane i = W_dw[c_grp_m × 16 + i, k_idx]
pe_a_is_vector ← 1
pe_op_mode ← MODE_MUL
```

PE Array tại posedge: mỗi lane `i`: `mult_i = x_inner[c_grp_m×16+i, t-3+k_idx]
× W_dw[c_grp_m×16+i, k_idx]`. Accumulator giữ.

Sau 4 lần lặp k_idx = 0..3, mỗi lane đã accumulate Σ x × W. Sau đó:

```
S_M2_WRITE:
    m_we      ← 1
    m_wr_addr ← B_X_CONV + t_stride_m + c_grp_m
    m_wr_data: each lane ← sat_add16(pe_out[lane], c_rd_data[lane])  // + bias
    bank_sel  ← 0  // write ram_b
```

bias từ ram_const ở C_M_DW_BIAS region.

#### M2 cycle cost: 4 × (3 substep) × 8 c_grp_m × T × 1.

Block 0: 4 × 3 × 8 × 1000 ≈ 96K cycle.

### 5.5 M3: SiLU activation

#### PyTorch

```python
u = F.silu(x_conv)
```

SiLU(x) = x × σ(x), x ∈ R, x_conv là Q4.11.

#### FPGA — LUT lookup

```
S_M3_MAC:
    m_rd_addr ← B_X_CONV + t_stride_m + c_grp_m
    
S_M3_WAIT, S_M3_LATCH:
    silu_in[0..15] ← m_rd_data[0..15 × 16 +: 16]   (assigned combinationally)
    silu_o[0..15] are combinational outputs of Silu_LUT instances
    
S_M3_WRITE:
    m_we      ← 1
    bank_sel  ← 1 (write ram_a)
    m_wr_addr ← A_X_INNER + t_stride_m + c_grp_m
    m_wr_data ← {silu_o[15], silu_o[14], ..., silu_o[0]}
```

Mỗi Silu_LUT instance là combinational ROM 256-entry × 16-bit. Index tính:
```
idx = (x_in + 16384) >>> 7    // map [-8, +8) → [0, 256)
silu_out = silu_table[idx]    // pre-computed at synth time
```

Wait — đoạn này đang nhầm input width. Để chính xác: `LUT_LO = -(1 << (FB+3))
= -16384` cho FB=11. Then `LUT_SHIFT = FB - 4 = 7`. Index = `(x_in - LUT_LO)
>>> LUT_SHIFT` = `(x + 16384) >> 7` ∈ [0, 256).

Out of range: x < -8 → 0, x ≥ +8 → x (silu(x) → x for large x).

### 5.6 M3CP: copy A_X_INNER → B_U_SAFE

PyTorch không có thao tác này — đây là **artifact của memory layout FPGA**.
Lý do: M5 sẽ ghi delta vào A_X_INNER, đè u; nhưng M6A, M6B sau đó vẫn cần u
để tính `dB × u` và `D × u`. Vì vậy phải copy u ra một region khác (B_U_SAFE)
trước khi M5 chạy.

```
S_M3CP_READ: m_rd_addr ← A_X_INNER + ...
S_M3CP_WAIT: wait
S_M3CP_LATCH: m_wr_data ← m_rd_data
S_M3CP_WRITE: bank_sel ← 0, m_wr_addr ← B_U_SAFE + ..., m_we ← 1
```

Pure copy, no compute. Cost: ~5 × CH_M × T = 5 × 8 × 1000 = 40K cycle cho
block 0.

(Note: RU-1 optimization plan đề xuất loại bỏ M3CP bằng cách đổi M3 destination,
nhưng chưa implement.)

### 5.7 M4: x_proj linear (d_inner → dt_rank + 2 × d_state)

#### PyTorch

```python
self.x_proj = nn.Linear(d_inner=128, dt_rank + 2 × d_state, bias=False)
# Output: vector chiều (dt_rank + 2 × d_state) per timestep
# Cho block 0: dt_rank=4, d_state=16 → output dim = 4 + 32 = 36
# Pad lên 48 (= 3 × 16) cho phân chia 16-lane.
# Output split: [delta_raw (4), B_raw (16), C_raw (16)]
```

#### FPGA

x_proj tổ chức weight thành 3 group × 16 lane:
- Group 0: W cho delta (4 valid + 12 zero pad)
- Group 1: W cho B (16 lane)
- Group 2: W cho C (16 lane)

MAC qua d_inner = 128 input channels, 3 group output:

```
for t in 0..T-1:
  for c_grp_out_xp in 0..2:   # 3 nhóm
    for mac_idx in 0..127:
      MAC tương tự P1, với pe_A = u_safe[mac_idx, t], pe_B = W_xp[c_grp×16+lane, mac_idx]
    write xproj_out[c_grp_out_xp × 16..+15, t] to B_X_CONV + t × 3 + c_grp_out_xp
```

**B_X_CONV được reuse** ở đây cho x_proj output (vì x_conv đã consumed ở M3).
Compact memory map cho phép.

Cycle M4 / timestep: 3 × (128 × 3 + 3) = 1161 cycle. Block 0 total: ~1.16M.

### 5.8 M5: dt_proj + bias + softplus

#### PyTorch

```python
self.dt_proj = nn.Linear(dt_rank=4, d_inner=128)
delta_raw = ...                         # (B, T, 4) từ x_proj
delta = F.softplus(dt_proj(delta_raw) + dt_proj.bias)   # (B, T, 128)
```

#### FPGA

Linear 4 → 128 per timestep, sau đó softplus per element.

```
for t in 0..T-1:
  for c_grp_m in 0..7:
    for mac_idx in 0..3:    # dt_rank = 4 input
      MAC với pe_A = delta_raw[mac_idx, t], pe_B = W_dt[c_grp_m × 16 + lane, mac_idx]
    # PE_out = linear projected delta (16 channel × 16-bit, registered)
    # Apply softplus per lane via sp LUT
    sp_in[0..15] ← sat_add16(pe_out[lane], bias[lane])
    sp_o[0..15] ← Softplus_LUT(sp_in[lane])
    
S_M5_WRITE:
    m_wr_addr ← A_X_INNER + t_stride_m + c_grp_m    # GHI ĐÈ x_inner / silu
    m_wr_data ← {sp_o[15..0]}
    bank_sel  ← 1
```

→ Sau M5, A_X_INNER chứa delta thay vì u. Đó là lý do M3CP đã copy u sang
B_U_SAFE trước đó.

Cycle M5 / timestep: ~50. Block 0 total: ~50K.

### 5.9 M6A: tính dA = exp(delta_t × A) per channel + read B

#### PyTorch (concept)

```python
# A = -exp(A_log), shape (d_inner, d_state) = (128, 16)
# delta = từ M5, shape (B, T, d_inner) = (1, T, 128)
# Per (t, c, s): dA[t, c, s] = exp(delta[t, c] × A[c, s])
# Per (t, s):    B[t, s]  = x_proj output (đã có ở B_X_CONV)
```

#### FPGA

S_M6A có 16 substep × s_idx (cho mỗi state dimension). Mỗi substep:

```
S_M6A_DA_READ: read A[c_grp_m × 16..+15, s_idx], read delta[c_grp_m × 16..+15, t]
S_M6A_DA_COMP: tính delta × A per lane qua PE Array MUL mode (a_is_vector=1)
S_M6A_DA_LUT:  exp_in[lane] = sat result; exp_o[lane] = Exp_LUT(exp_in[lane])
S_M6A_DA_LATCH: dA_reg[lane] ← exp_o[lane]
S_M6A_T2_READ: read u_safe[c_grp_m × 16..+15, t] from B_U_SAFE
S_M6A_T2_LATCH: u_scalar_reg[lane] ← u
S_M6A_DB_READ: read B[s_idx, t] (scalar) → B_scalar_reg
S_M6A_DB_LATCH: term2_reg[lane] = (B_scalar_reg × delta[lane]) × u
                                  (per lane MUL combinational)
```

Đây là phần phức tạp nhất Mamba. Tổng cycle / (t, c_grp_m, s_idx): ~15 cycle.

### 5.10 M6B: scan h_t = dA × h_{t-1} + dB × u + (cập nhật h_reg)

#### PyTorch

```python
# h_0 = 0; for t in 0..T-1: for c, s: h[t, c, s] = dA[t,c,s] × h[t-1,c,s] + dB[t,c,s] × u[t,c]
# Đây là vòng lặp sequential (h phụ thuộc h trước)
```

#### FPGA

State `h_reg` (256-bit) lưu hidden state cho 16 lane (= 1 c_grp_m × 16 state
dim). Vòng lặp:

```
for t in 0..T-1:
  for c_grp_m in 0..7:
    h_reg ← 0  (chỉ ở t=0)
    for s_idx in 0..15:
      đọc dA_reg, dB×u từ M6A
      h_reg[lane] ← sat_add16(dA_reg[lane] × h_reg[lane], term2_reg[lane])
        (PE Array MUL + ADD; 1 cycle reduction per lane)
      write h_reg[lane] to A_H_STATE[c_grp_m × 16 + s_idx]
```

Cycle / (t, c_grp_m, s_idx): ~17. Block 0: 1000 × 8 × 16 × 17 ≈ 2.17M cycle —
**lớn nhất trong Mamba**.

### 5.11 M7: y_gated = (C × h + D × u) × SiLU(z_gate)

PyTorch:
```python
y = (h × C^T).sum(dim=state) + D × u    # SSM output
y_gated = y × silu(z)
```

FPGA: tương tự M6B nhưng read C, h, D, u → tính y → silu(z) → multiply. Cycle
cao do nhiều substep.

### 5.12 M8: out_proj (linear d_inner → d_in = 64)

Cuối Mamba: project y_gated từ 128 channel xuống 64.

```python
mamba_out = self.out_proj(y_gated)
```

FPGA MAC giống M1A nhưng input d_inner = 128, output d_in = 64. Ghi vào
A_MAMBA_OUT.

Cycle M8 / t: ~800. Block 0 total: ~800K cycle.

### 5.13 Tổng cycle phase Mamba block 0

| Sub-phase | Cycle (block 0, T=1000) |
|---|---|
| NORM + M1A | 1.6M |
| NORM + M1B | 1.6M |
| M2 | 96K |
| M3 | 40K |
| M3CP | 40K |
| M4 | 1.16M |
| M5 | 50K |
| M6A | 1.9M |
| M6B | 2.17M |
| M7 | 1.1M |
| M8 | 800K |
| **Total** | **~10.5M cycle** |

Sau khi M8 xong:
- A_MAMBA_OUT (ram_a[8000..11999]) chứa mamba_out_q[0..63, 0..999]
- done_mamba ← 1
- state → S_FIN_READ

---

## 6. Phase 4 — FIN (Final stage Block 0)

### 6.1 PyTorch

```python
# inception_out (cat 4 branches + BN + ReLU) đã trong incep_cat_q
x1 = self.relu(self.bn(incep_cat))   # bn_relu(inc) per channel
x2 = self.relu(mamba_out)            # relu(mamba_out)
out = x1 + x2                        # sat_add
```

`bn_relu` per channel:
```
bn_out = sat16(sat16(inc × scale >> FB) + shift)
x1     = relu(bn_out) = max(bn_out, 0)
```

### 6.2 FPGA — flow

```
for t in 0..T-1:
  for c_grp_out in 0..3:   # 4 nhóm × 16 channel = 64 output
    S_FIN_READ:    đọc inception output cho c_grp_out (từ A_CH1, B_CH2, B_CH3, B_CH4
                   tùy fin_branch)
    S_FIN_MUL:     đọc Inc BN scale, shift; cache incep_reg ← m_rd_data
    S_FIN_WAIT2:   đọc mamba_out (cùng địa chỉ ở bank A)
    S_FIN_WRITE:   ghi m_wr_data, mỗi lane = sat_add16(bn_relu(incep, scale, shift),
                                                        relu16(mamba))
                   m_wr_addr = (c_grp == 0) ? B_FINAL_OUT + t_stride_out
                                            : A_FINAL_OUT + t_stride_out + c_grp
                   bank_sel = (c_grp == 0) ? 0 : 1
  S_FIN_NEXT: advance t hoặc c_grp; cuối cùng → CASCADE/HEAD/DONE
```

### 6.3 Cycle FIN block 0: ~10K (negligible)

### 6.4 Memory sau FIN

- `B_FINAL_OUT[t × 4]` (c_grp=0): block 0 output cho channel 0..15 mỗi t
- `A_FINAL_OUT[t × 4 + c_grp]` (c_grp=1..3): output cho channel 16..63

---

## 7. Phase 5 — Cascade Block 0 → Block 1

### 7.1 PyTorch

Block 0 output (1, 64, 1000) trực tiếp là input của Block 1 (không qua maxpool
vì giữa block 0 và 1 không có maxpool).

### 7.2 FPGA — copy FINAL_OUT → A_INPUT_BASE

```
S_CASCADE_RA:  bank_sel ← (c_grp == 0) ? 1 : 0
               m_rd_addr ← (c_grp == 0 ? B_FINAL_OUT : A_FINAL_OUT) +
                           src_t_a × CH_OUT + c_grp
S_CASCADE_WA:  wait
S_CASCADE_RB:  latch m_rd_data → max_buf
               nếu need_pool: set m_rd_addr cho src_t_b
S_CASCADE_WB:  wait
S_CASCADE_WR:  m_we ← 1; bank_sel ← 1 (write ram_a)
               m_wr_addr ← A_INPUT_BASE + t_out × CH_OUT + c_grp
               m_wr_data ← need_pool ? elem_max16(max_buf, m_rd_data) : max_buf
               advance t_out; cuối cùng done_all ← 1
```

Block 0 → 1: cascade_mode = 1, need_pool = 0 → copy mode.

Cost: 4 c_grp × 1000 t × 5 cycle ≈ 20K cycle.

Sau cascade, ram_a[0..3999] chứa block 0 output, sẵn sàng làm input cho Block 1.

### 7.3 done_all pulse + chờ start mới

Host detect `done_all = 1`, DMA-load block 1 weights, pulse `start`. Lần này:
- enc_mode = 0 (encoder đã chạy lần đầu)
- cascade_mode = 1, need_pool = 0 (block 1 → 2 không cần pool — nhưng block 1
  → 2 phải pool!)

**Sửa**: cascade/need_pool áp dụng KHI block hiện tại xong (chứ không phải khi
block sau chạy). Vậy:
- Block 0 (đang chạy): cascade=1, need_pool=0 → sau xong copy thẳng cho block 1
- Block 1 (sẽ chạy sau): cascade=1, need_pool=1 → sau xong pool stride-2 cho block 2
- Block 2: cascade=1, need_pool=0
- Block 3: cascade=1, need_pool=1 → pool cho block 4
- Block 4: cascade=0, head_mode=1

---

## 8. Block 1 — lặp giống Block 0

Sau cascade block 0, host trigger Block 1 với cùng config (CH_IN=CH_OUT=4,
CH_M=8, DT_RANK=4, T=1000). Block 1 weights mới được DMA-load.

Flow giống Block 0:
- S_IDLE start → enc_mode=0 → S_P1_MAC trực tiếp
- P1 → BR → NORM/Mamba → FIN
- Cuối FIN: cascade_mode=1, need_pool=1 → S_CASCADE với pool

Cycle Block 1 ≈ Block 0 = ~15.5M cycle.

### 8.1 Cascade Block 1 → 2 với pool

`need_pool_reg = 1`. Cascade FSM đọc 2 timestep liên tiếp, lấy max:

```
S_CASCADE_RA (t_out): m_rd_addr ← src for src_t_a = 2 × t_out
S_CASCADE_RB:         max_buf ← m_rd_data;
                      m_rd_addr ← src for src_t_b = 2 × t_out + 1
S_CASCADE_WR:         m_wr_data ← elem_max16(max_buf, m_rd_data) (16 lane max)
                      m_wr_addr ← A_INPUT_BASE + t_out × 4 + c_grp
```

Sau pool, A_INPUT_BASE chứa 500 timestep (thay vì 1000).

---

## 9. Block 2, 3 — T = 500

Tương tự Block 0/1 nhưng T = 500. Host set T_MAX = 500 trước start.

Cycle / block 2: ~7.5M.
Cycle / block 3: ~7.5M.

Cascade Block 3 → 4 với pool → T = 250.

---

## 10. Block 4 — T = 250, d_out = 128, d_inner = 256

### 10.1 Khác biệt config

- CH_OUT = 8 → d_out = 128 (gấp đôi)
- CH_M = 16 → d_inner = 256 (gấp đôi)
- DT_RANK = 8
- T = 250

→ Inception có thêm `c_grp_br` loop (br_grp_last = 1, br_dim_groups = 2). MAC
qua dim = 32 thay vì 16. Mamba scan qua 16 c_grp_m thay vì 8.

### 10.2 Cycle Block 4

| Phase | Cycle |
|---|---|
| Phase 1 | 390K |
| Inception | 3.86M |
| Mamba | 7.43M |
| FIN | 16K |
| **Total** | **11.7M** |

Cuối FIN với `cascade_mode=0, head_mode=1`:
```
S_FIN_NEXT khi c_grp == ch_out_last (= 7) và t_cnt == t_last (= 249):
    if (cascade_mode_reg) { ... }
    else if (head_mode_reg) {
        gap_c_grp ← 0; gap_t ← 0
        for i in 0..15: gap_sum[0..7][i] ← 0
        state ← S_GAP_READ
    }
```

→ Chuyển sang GAP.

---

## 11. Phase 6 — GAP (Global Average Pooling)

### 11.1 PyTorch

```python
x = x.mean(dim=-1)   # (B, 128, T=250) → (B, 128)
```

Per channel `c`: `gap[c] = (1/T) × Σ_{t=0..T-1} final[c, t]`

### 11.2 FPGA — sum + multiply by reciprocal

```
for c_grp_in 0..7:
  for t_gap 0..249:
    bank_sel ← (c_grp_in == 0) ? 1 : 0   # đọc B_FINAL_OUT hoặc A_FINAL_OUT
    m_rd_addr ← base + t_gap × CH_OUT + c_grp_in
    đợi BRAM read 1 cycle
    for each lane i in 0..15:
      gap_sum[c_grp_in][i] += sign_extend(m_rd_data[i × 16 +: 16] → 24-bit)
```

Sau khi hoàn thành accumulate (8 × 250 = 2000 cycle MAC + ~1000 cycle wait =
~3000 cycle), finalize:

```
for c_grp 0..7:
  for lane 0..15:
    gap_q_reg[c_grp][lane × 16 +: 16] ← sat_mul_q15(gap_sum[c_grp][lane])
    # sat_mul_q15 = sat16( gap_sum × INV_T_Q15 >> 15 )
    # INV_T_Q15 = 131 (≈ 2^15 / 250)
```

`sat_mul_q15` per lane: 1 cycle combinational (1 DSP mul 24×16 → 40-bit, shift
>> 15, sat16). 8 cycle cho 8 c_grp.

Sau GAP: `gap_q_reg[0..7]` chứa 128 GAP values (16 lane × 8 word).

### 11.3 Trace S_GAP_LATCH chi tiết (gap_c_grp=0, gap_t=0)

```
m_rd_data tại cycle này = ram_b[B_FINAL_OUT + 0 × 8 + 0] = block4_final[0..15, 0]
gap_sum[0][0] ← gap_sum[0][0] + sign_extend(m_rd_data[0 +: 16], 24)
gap_sum[0][1] ← gap_sum[0][1] + sign_extend(m_rd_data[16 +: 16], 24)
...
gap_sum[0][15] ← gap_sum[0][15] + sign_extend(m_rd_data[240 +: 16], 24)
state ← S_GAP_NEXT
```

Combinational 16-input add per lane: nhẹ, không là CP.

### 11.4 done_gap pulse

Cuối S_GAP_FIN_NEXT (sau khi gap_q_reg cuối cùng được tính):

```
done_gap ← 1
fc_class ← 0; fc_grp_in ← 0; fc_lane ← 0; fc_acc ← 0
c_rd_addr ← C_FC_BIAS    // = 68
state ← S_FC_LOAD_BIAS
```

---

## 12. Phase 7 — FC (Classifier)

### 12.1 PyTorch

```python
self.classifier = nn.Linear(2 × d_model = 128, n_classes = 5, bias=True)
logits = self.classifier(gap)
# logits[c] = Σ_i W_fc[c, i] × gap[i] + b_fc[c]
```

### 12.2 FPGA — lane-serial MAC (sau fix timing)

#### S_FC_LOAD_BIAS → S_FC_BIAS_WAIT

```
S_FC_LOAD_BIAS:    state ← S_FC_BIAS_WAIT  (1 cycle wait BRAM)
S_FC_BIAS_WAIT:
    fc_bias_lane[0] ← c_rd_data[0 +: 16]    // bias class 0
    fc_bias_lane[1] ← c_rd_data[16 +: 16]   // bias class 1
    ...
    fc_bias_lane[4] ← c_rd_data[64 +: 16]   // bias class 4
    w_rd_addr ← W_FC_BASE + fc_class × 8 + fc_grp_in = 14064 + 0 + 0 = 14064
    fc_acc ← 0
    substep ← 0
    state ← S_FC_MAC
```

#### S_FC_MAC chi tiết (fc_class=0, fc_grp_in=0)

```
substep 0:
    w_rd_addr ← W_FC_BASE + 0 × 8 + 0 = 14064
    fc_gap_word ← gap_q_reg[fc_grp_in] = gap_q_reg[0]
    fc_lane ← 0
    substep ← 1

substep 1: wait BRAM 1 cycle
    w_rd_data sẽ valid ở substep 2 đầu

substep 2 (lane 0):
    fc_acc ← fc_acc + (w_rd_data[0 +: 16] × fc_gap_word[0 +: 16])
    fc_lane ← 1
    (stay at substep 2)

substep 2 (lane 1):
    fc_acc ← fc_acc + (w_rd_data[16 +: 16] × fc_gap_word[16 +: 16])
    fc_lane ← 2
...

substep 2 (lane 15):
    fc_acc ← fc_acc + (w_rd_data[240 +: 16] × fc_gap_word[240 +: 16])
    fc_lane ← 0
    if (fc_grp_in == 7) state ← S_FC_WAIT
    else { fc_grp_in ← 1; substep ← 0 }
```

Sau 16 cycle ở substep 2, fc_acc accumulated 16 product. Total per grp_in =
18 cycle (2 wait + 16 MAC). Per class = 8 × 18 = 144 cycle. Per FC phase =
5 × 144 = 720 cycle.

#### S_FC_NEXT_CLASS

```
case (fc_class)
    0: logit0 ← sat_add16(sat16(fc_acc >>> 11), fc_bias_lane[0])
    1: logit1 ← sat_add16(sat16(fc_acc >>> 11), fc_bias_lane[1])
    ... 
endcase
if (fc_class == 4) state ← S_FC_FINALIZE
else {
    fc_class ← fc_class + 1
    fc_grp_in ← 0
    fc_acc ← 0
    substep ← 0
    state ← S_FC_MAC
}
```

#### S_FC_FINALIZE → S_FC_DONE → S_IDLE

```
S_FC_FINALIZE: done_fc ← 1; state ← S_FC_DONE
S_FC_DONE: done_phase1 ← 1; done_inception ← 1; done_mamba ← 1; done_all ← 1
           state ← S_IDLE
```

Tại đây `logit0..4` đã chứa 5 giá trị Q4.11 cuối cùng. Host detect `done_fc`
và đọc.

---

## 13. Output: Argmax và class prediction

### 13.1 PyTorch

```python
pred = torch.argmax(logits, dim=-1).item()
```

### 13.2 FPGA / Host

`logit0..4` (5 × 16-bit signed) là output port của top-level. Host (CPU)
đọc 5 values, so sánh:

```
max_l = logit0; pred = 0
if (logit1 > max_l) { max_l = logit1; pred = 1 }
if (logit2 > max_l) { max_l = logit2; pred = 2 }
if (logit3 > max_l) { max_l = logit3; pred = 3 }
if (logit4 > max_l) { max_l = logit4; pred = 4 }
return pred
```

→ `pred ∈ {0, 1, 2, 3, 4}` là class predict cuối cùng.

Đối với PTB-XL super-class:
- 0 = NORM (Normal)
- 1 = MI (Myocardial Infarction)
- 2 = STTC (ST/T Change)
- 3 = CD (Conduction Disturbance)
- 4 = HYP (Hypertrophy)

---

## 14. Tổng kết end-to-end cho 1 sample

### 14.1 Cycle count tổng

| Phase | Cycle | Time @ 100 MHz |
|---|---|---|
| Encoder | 64K | 0.64 ms |
| Block 0 (P1+BR+Mam+FIN+Cascade) | 15.5M | 155 ms |
| Block 1 | 15.5M | 155 ms |
| Block 2 (T=500) | 7.7M | 77 ms |
| Block 3 (T=500) | 7.7M | 77 ms |
| Block 4 (T=250) | 11.7M | 117 ms |
| GAP | 6K | 0.06 ms |
| FC | 720 | 0.0072 ms |
| **Tổng** | **~58M** | **~580 ms** |

DMA load overhead (load 5 block × ~13K word weight + waveform + encoder/FC
weight first-time):
- Block weight: ~13K word × 5 × 2 cycle/word = 130K cycle ≈ 1.3 ms × 5 =
  6.5 ms tổng
- Waveform: 1000 word × 2 cycle = 2K cycle = 0.02 ms
- Encoder/FC weight first time: ~108 word × 2 = 216 cycle = 0.002 ms

**Total per sample @ 100 MHz**: ~590 ms (compute + DMA).

### 14.2 Memory access pattern trong 1 sample

| Region | Lần read | Lần write |
|---|---|---|
| ram_a (40 URAM) | ~12M (all phase) | ~5M |
| ram_b (40 URAM) | ~10M | ~4M |
| ram_weight (118 BRAM) | ~30M | 0 (chỉ DMA) |
| ram_const (LUT) | ~5M | 0 |

### 14.3 DSP utilization

- **16 DSP** ở PE_Array dùng liên tục trong các phase MAC (P1, BR, M1A/B, M2,
  M4, M5, M6A/B, M7, M8) — duty cycle ~70%
- **1 DSP** ở RMSNorm_Mul cho RMSNorm output
- **1 DSP** trong FC reduce (sau timing fix)
- **Các DSP khác** rải rác trong sat/bn_relu chains (~40)
- Tổng ~60 DSP.

### 14.4 PyTorch ↔ FPGA equivalence summary

| PyTorch op | FPGA implementation | State |
|---|---|---|
| `Conv1d(12,64,k=1) + BN` | MAC 12 inputs + bias, 4 c_grp_out | S_ENC_* |
| `Conv1d(64,64,k=1) + BN` block conv | MAC 64 inputs + bias, 4 c_grp_out | S_P1_* |
| `Conv1d(d_out, d_out/4, k=1)` bottleneck | MAC 64 inputs, no bias | S_BR_MAC branch_id=0 |
| `MaxPool(k=3) + Conv1d` B1 | 3-cycle max + MAC | S_BR_MAC branch_id=1 |
| `Conv1d(dim, dim, k=9/19/39)` B2/3/4 | k×16-input MAC qua kernel position | S_BR_MAC branch_id=2..4 |
| `RMSNorm` | norm_sq + RSqrt ROM + RMSNorm_Mul | S_NORM_M1*_* |
| `Linear(64, 128) ×2` in_proj | MAC qua 64 input, 8 c_grp_out | S_M1A_*, S_M1B_* |
| `Conv1d(128,128,k=4, groups=128)` depthwise | element-wise mul (PE MUL mode) × 4 k | S_M2_* |
| `SiLU` | LUT lookup 16-lane | S_M3_* |
| (mem layout artifact) | copy u_safe | S_M3CP_* |
| `Linear(128, 36)` x_proj | MAC qua 128, 3 c_grp_xp | S_M4_* |
| `Linear(4, 128) + bias + softplus` | MAC + LUT | S_M5_* |
| `exp(δ·A)` | MUL + Exp_LUT | S_M6A_DA_* |
| SSM scan `h = dA·h + dB·u` | MUL + sat_add | S_M6B_* |
| `C·h + D·u` y_gated × silu(z) | MUL + sat + LUT | S_M7_* |
| `Linear(128, 64)` out_proj | MAC qua 128 | S_M8_* |
| `BN+ReLU(inc) + ReLU(mam)` final | bn_relu + relu + sat_add | S_FIN_WRITE |
| copy/pool cho block kế | cascade FSM | S_CASCADE_* |
| `mean(dim=-1)` GAP | accumulator + sat_mul_q15 | S_GAP_* |
| `Linear(128, 5)` classifier | lane-serial MAC + bias | S_FC_* |
| `argmax(logits)` | host CPU | — |

---

## 15. Phụ lục — Mapping cụ thể 1 phép tính từ float → integer

### 15.1 Ví dụ: encoder output channel 0, timestep 0

#### Float (PyTorch)

```
W_conv[0, :, 0] = [0.1, -0.2, 0.05, ..., -0.15]  # 12 weights
b_conv[0] = 0.3
γ_BN[0] = 1.2
β_BN[0] = 0.05
μ_BN[0] = 0.1
σ_BN[0] = 1.5

W_fused[0, j] = W_conv[0, j, 0] × γ / σ
              ≈ [0.08, -0.16, 0.04, ..., -0.12]
b_fused[0] = (b_conv[0] - μ) × γ / σ + β
           ≈ 0.21

x[0..11, 0] = [-0.5, 0.3, -0.1, 0.8, ..., -0.2]  # 12 leads ở t=0

enc[0, 0] = Σ W_fused[0, j] × x[j, 0] + b_fused[0]
          ≈ 0.45
```

#### Integer (FPGA Q4.11)

```
W_q[0, j]    = floor(W_fused[0, j] × 2048) ∈ {164, -328, 82, ..., -246}
b_q[0]       = floor(0.21 × 2048) = 430

X_q[j, 0]    = floor(x[j, 0] × 2048) ∈ {-1024, 614, -204, 1638, ..., -409}

(in PE Array, j=0..11, cycle theo cycle:)
acc_raw_0 = X_q[0,0] × W_q[0,0]         = -1024 × 164    = -167,936
acc_raw_0 += X_q[1,0] × W_q[0,1]         = 614 × -328    = -201,392
acc_raw_0 += X_q[2,0] × W_q[0,2]         = -204 × 82      = -16,728
...                                                       (cộng dồn 12 lần)
acc_raw_0 (sau 12 MAC) ≈ ~920,000  (40-bit signed)

# Substep 2 last MAC mac_idx=11:
out_val_0 = sat16(acc_raw_0 >>> 11) = sat16(449) = 449

# S_ENC_WRITE: lane 0:
m_wr_data[0 +: 16] = sat_add16(pe_out[0], c_rd_data[0])
                   = sat_add16(449, 430)
                   = 879   (Q4.11) = 879/2048 ≈ 0.429

# So với float 0.45 → sai số ~0.02 (~ ±10 LSB), within tolerance.
```

→ Ngân hàng bộ nhớ ram_a tại offset 0 lưu word có lane 0 = `879`, lane 1..15
là các output channel 1..15 tương tự.

---

## 16. Note kết luận

Toàn bộ pipeline cho 1 sample thực hiện ~60 triệu phép MAC trên FPGA, qua
~147 FSM state, sử dụng 16 DSP cho PE_Array cộng các DSP/LUT/CARRY rải rác,
~580 ms latency. Mọi phép tính có **ánh xạ chính xác tới một block PyTorch
tương ứng** (đã verify byte-exact 100/100 sample qua TB v2). Sự sai số AUC
−0.0026 so với float64 reference đến hoàn toàn từ quantization Q4.11, không
phải từ design HW.

Mọi state nói trên đều **tuần tự**, không có parallelism giữa các phase
(encoder phải xong mới đến P1; M1A xong mới đến M1B; v.v.). Parallelism duy
nhất là **16-way lane** trong PE_Array. Đây là trade-off có chủ ý: thiết kế
tận dụng PE_Array 100% trong các phase MAC chính, đổi lấy cycle count cao
(~58M cycle/sample) nhưng resource thấp (60 DSP) và Fmax cao (100 MHz). Real
time per sample = 580 ms vẫn rất khả thi cho ECG offline diagnosis.
