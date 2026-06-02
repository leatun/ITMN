# ITMN Optimization — Notes & Directives

Sống văn bản này song song với `FMAX_RESOURCE_PLAN.md`. Plan kia mang tính tactical (fix nào, đâu, bao nhiêu cycle). File này lưu **định hướng cấp cao** từ giáo viên + observation của bản thân, dùng để định khung mọi quyết định optimize tiếp theo.

---

## 1. Goal tối ưu hệ thống

| # | Mục tiêu | Lý do |
|---|----------|-------|
| G1 | Đạt số liệu RTL **so sánh được với paper Mamba/RNN/CNN HW** | Để paper có chỗ đứng. Hiện tại số liệu chưa "fair compare" do thiết kế gom inception+mamba làm một |
| G2 | Tối ưu **Fmax** (giảm critical path) | Hiện 71.4 MHz với WNS chỉ 0.394ns — quá sát, không có headroom |
| G3 | Tối ưu **HW resource** (LUT, DSP, BRAM, URAM) | DSP đang khá cao so với paper đối chứng. Cần truy nguồn & giảm |
| G4 | Giữ **AUC + TPR** không giảm so với float | Hiện đã ổn (AUC 0.9328 vs float ≈ 0.9354) — chỉ cần bảo toàn khi sửa |
| G5 | Cấu trúc lại để **paper-friendly** (tách module, đặt tên rõ) | Liên quan G1 — phục vụ comparison & methodology section |

---

## 2. Gợi ý từ giáo viên (advisor directives)

### D1 — Tách bạch INCEPTION và MAMBA để báo cáo riêng

**Vấn đề**: Thiết kế hiện tại nhồi cả Inception (Phase BR: 4 conv branch) + Mamba (M1A..M8) + Final stage vào cùng 1 controller, dùng chung URAM + PE_Array + Activation LUT. Khi so sánh:
- Paper khác báo cáo `MambaHW` riêng → không biết bảng so sánh phải lấy số nào của bạn.
- Inception riêng cũng vậy.

**Cần làm**:
- Hoặc **tách thành 2 module riêng** (Inception_Block, Mamba_Block) — cho phép synth riêng và đo riêng. Nhược điểm: phải duplicate URAM/PE — chi phí HW thật.
- Hoặc **đo riêng phần FSM** thuộc về Inception (P1+BR+FIN không tính mamba) vs Mamba (M1..M8) — bằng cách:
  - Đếm cycle riêng cho từng phase trong TB (đã có sẵn `done_phase1`, `done_inception`, `done_mamba`).
  - Tách báo cáo resource: phần FSM/datapath chỉ phục vụ inception vs chỉ phục vụ mamba vs shared. Có thể làm bằng `keep_hierarchy` + per-module utilization report.

**Status**: chưa làm. Ưu tiên cao nếu định viết paper sớm.

---

### D2 — KHÔNG dùng Sharing Buffer Allocator (SBA)

**Bối cảnh**: Thiết kế ban đầu tham khảo paper của giáo viên dùng SBA để feed PE cho các kernel khác nhau (đặc biệt conv 9/19/39 trong inception).

**Lý do tránh**: SBA tạo **1 critical path lớn** — mux tập trung qua nhiều port làm cell delay cao + fanout lớn.

**Cần làm**:
- Audit thiết kế hiện tại xem có giống SBA không (hay đã khác bản chất). Hiện đang dùng PE_Array nhận `pe_A` scalar broadcast + `pe_B` 256-bit vector → có vẻ KHÔNG phải SBA truyền thống.
- Nếu vẫn tệ hơn cách giáo viên làm → đổi sang scheme khác (vd. dedicated per-kernel PE, hoặc per-branch reg-buffer thay vì mux tập trung).
- Đo: critical path **trong inception phase** có dính tới mux multi-port không?

**Status**: chưa audit kỹ. Cần phân tích sau khi xong CP-1/CP-2.

---

### D3 — Gộp `ram_const` vào Activation LUT + rsqrt ROM (LUT-based const storage)

**Vấn đề hiện tại**:
- `ram_const` (BRAM 64×256-bit) chứa các hằng số per-channel: P1 bias, BN scale/shift cho inception, depthwise conv bias, dt bias, RMSNorm gamma weights.
- Đồng thời có **3 activation LUT** (silu, softplus, exp) + **1 rsqrt ROM** sống riêng.

**Định hướng**: Tất cả "constant data" (= bias, gamma, LUT entries, rsqrt entries) nên **gộp chung** vào 1 LUT-based storage (distributed RAM hoặc dedicated ROM). Lý do:
- Giảm BRAM (giải phóng `ram_const` BRAM).
- Đồng nhất interface const-read trên FSM.
- Có thể chạy nhanh hơn nếu dùng LUT distributed (1 cycle, no register delay).

**Cần làm**:
- Map lại storage: bias/scale/shift/dt_bias/gamma → 1 vùng `const_LUT`; silu/softplus/exp/rsqrt → 1 vùng `act_LUT` (hoặc gộp chung tất cả). Kích thước tổng cần đo lại.
- Convert RAM_STYLE từ "block" → "distributed" hoặc dùng `(* rom_style = "distributed" *)`.

**Status**: chưa làm. Cần kiểm tra size tổng có vừa distributed LUT không (nếu quá lớn vẫn phải dùng BRAM, nhưng gộp 1 BRAM).

---

### D4 — Truy nguồn vì sao `ram_weight` ăn 114 BRAM nhưng `ram_a`/`ram_b` không, và LUT cost lệch nhau

**Quan sát**:
- `ram_weight` (BRAM_256b, ADDR_WIDTH=14, "block") → 114 BRAM, 1201 LUT
- `ram_const` (BRAM_256b, ADDR_WIDTH=6,  "block") → 0 BRAM (?), 724 LUT
- `ram_a` (BRAM_256b, ADDR_WIDTH=15, "ultra") → 0 BRAM, **317 LUT**, dùng URAM
- `ram_b` (BRAM_256b, ADDR_WIDTH=15, "ultra") → 0 BRAM, **3105 LUT**, dùng URAM

**Câu hỏi**:
- Sao `ram_b` tốn 3105 LUT trong khi `ram_a` chỉ 317 LUT, dù cùng spec?
- Có phải Vivado "attribute" combinational logic của downstream consumer (mux 16-to-1 dữ liệu, `x_norm_fn`, `bn_relu`, activation input) vào hierarchy `ram_b`?
- Tại sao `ram_const` 64×256 lại được implement bằng distributed LUT (724 LUT) chứ không phải BRAM, dù khai báo `"block"`?

**Cần làm**: 
- Mở schematic Vivado, navigate `mem_sys/ram_b` → expand → xem actual primitive (LUT thật vs URAM cascade).
- Kiểm tra synthesis report xem có warning kiểu "RAM_STYLE=block but inferred distributed" không.
- Nếu xác nhận Vivado attribute LUT từ consumer → không phải bug, chỉ là hierarchical reporting quirk.
- Đây cũng là hint cho D3: nếu Vivado đã tự convert ram_const → distributed thì lý do gộp với activation LUT càng rõ.

**Status**: chưa truy. Liên quan trực tiếp tới D3.

---

## 3. Observation cá nhân

### O1 — Hệ thống đã ổn định functional

- Quantize chain (extract → test_hw → RTL) byte-exact qua `verify_byte_exact.py` (5/5 block PASS).
- AUC 0.9328 / float 0.9354 → gap −0.003 (chấp nhận được).
- TPR ổn. Đã verify xsim TB pass với golden.

→ **Mọi optimization tiếp theo chỉ được phép là functional no-op** (cùng arithmetic, chỉ thay đổi pipeline depth / resource sharing / addressing). Bất kỳ thay đổi giá trị (quantization scheme, formula) phải re-verify cả AUC.

### O2 — Open question: có cần encoder + fully-connected layer trên FPGA không?

**Hiện trạng**: Thiết kế RTL **không có** encoder (input projection) và FC layer cuối (classifier). Host CPU/Python xử lý 2 phần này, RTL chỉ chạy 5 ITM block.

**Câu hỏi**:
- Encoder thường rất nhỏ (vài chục param) — có nên gắn vào RTL để inference end-to-end?
- FC final layer kích thước nhỏ — có nên gắn?
- Nếu gắn → throughput tăng (không cần round-trip host), nhưng resource tăng.
- Nếu không gắn → giữ minimal, paper báo "ITM-block accelerator" không phải end-to-end → tùy framing.

**Cần quyết định trước khi viết paper**: scope là "ITM-block kernel" hay "end-to-end ECG classifier"?

### O3 — Adder rải rác: nên có 1 block `adder_unit` dùng chung?

**Hiện trạng**: Khắp controller có `sat_add16`, `sat_add`, 17-bit adders trong `bn_relu`, 16-input adder tree trong `norm_sq16_fn`, các adder address compute, accumulator trong PE. Tổng cộng có vài chục chỗ "ad-hoc" adder.

**Câu hỏi**:
- Có nên gộp lại thành 1 (hoặc vài) `adder_unit` time-multiplex, route data qua đó?
- Hay giữ như hiện tại (distributed) vì adder rẻ và share sẽ tốn mux đắt hơn?

**Phân tích sơ bộ**:
- Adder 16-bit ≈ 16 LUT (CARRY8) → rẻ. Share gây mux 256-bit → đắt + thêm critical path. **Có vẻ không lợi**.
- Trừ khi adder TREE lớn (vd 16-input 40-bit của `norm_sq16_fn`) — đó là chỗ thật sự đáng pipeline (CP-2).

**Hướng**: KHÔNG gom adder thường, NHƯNG **pipeline các adder tree lớn** (đã có trong CP-2). Đó là "share thông qua time" hợp lý hơn share thông qua mux.

---

## 4. Quyết định đã chốt (Q1-Q4)

| ID | Câu hỏi | **Quyết định** | Hệ quả |
|----|---------|---------------|--------|
| **Q1** | Target Fmax design | **125 MHz / period 8 ns** | CP-1 pipeline **1-stage giữa 2 mult** trong x_norm_fn là đủ. Không cần aggressive multi-stage. XDC sẽ tighten từ 14ns → 8ns sau khi xong CP-1..CP-3 |
| **Q2** | Cycle overhead budget per fix | **Không giới hạn** — ưu tiên Fmax đạt mục tiêu | Cho phép pipeline sâu nếu cần. Mỗi fix vẫn nên thiết kế hợp lý (không waste cycle vô cớ), nhưng không reject vì cycle cost |
| **Q3** | Paper scope (kernel-only vs end-to-end) | **Để mở** — quyết sau | Tạm bỏ qua D1/O2. Tactical fix CP-1..CP-3 trước. Khi gần viết paper sẽ chọn |
| **Q4** | Verification cadence | **Sau mỗi CP-fix**: xsim TB + verify_byte_exact | Mỗi fix là 1 step độc lập, verify trước khi sang fix kế tiếp. Phát hiện regression sớm |

**Hệ quả workflow cho CP-1**:
1. Implement CP-1 trong `ITM_CONTROLLER.v` (M1A_MAC + M1B_MAC + thêm register intermediate)
2. Chạy xsim TB → xác nhận PASS
3. Chạy verify_byte_exact.py → xác nhận no AUC change
4. Re-synth → check Fmax (kỳ vọng đạt ~125 MHz hoặc tốt hơn)
5. Đo lại cycle count, tính throughput mới
6. → Quyết tiếp CP-2 hay CP-3 dựa trên path tệ nhất mới

---

## 5. Câu hỏi mở cần cân nhắc trước/trong khi tackle CP-1..CP-3

| ID | Câu hỏi | Khi nào quan trọng | Ghi chú |
|----|---------|-------------------|---------|
| **Q5** | Pipeline CP-1: chèn register sau lần mult ĐẦU (trước sat) hay sau sat ĐẦU (trước mult thứ 2)? | Khi implement CP-1 | "Sau sat đầu" sạch hơn semantically (`p1_norm_reg` là Q4.11 16-bit). "Sau mult đầu" giữ 32-bit chính xác hơn — nhưng cần thêm reg width và rebalance shift. **Đề xuất**: sau sat đầu — giữ 16-bit, byte-exact với code hiện tại |
| **Q6** | Block 4 (CH_M=16, d_inner=256, T=250) có cần xử lý đặc biệt cho CP-1 không? | Khi implement CP-1 | CP-1 fix chỉ thay đổi inner loop substep, **không** phụ thuộc d_inner/T. Hoạt động giống nhau cho mọi block. ✅ Không cần special-case |
| **Q7** | Khi pipeline xong CP-1, critical path sẽ shift sang đâu? | Sau CP-1 verify | Khả năng cao: (a) `norm_sq16_fn` 16-input adder tree (CP-2), hoặc (b) `S_FIN_WRITE` 16-lane bn_relu chain (CP-3). Phải re-run synth + report timing để xác định path thứ 2 |
| **Q8** | Có dùng Vivado **retiming** auto (`-retiming`) trước khi manual pipeline? | Trước CP-1 implement | Retiming có thể tự move register đã có quanh combinational logic → đôi khi không cần manual fix. **Đề xuất**: thử retiming switch trước → đo Fmax. Nếu đã đạt 8ns mà không cần code change → tuyệt. Nếu chưa → mới manual CP-1 |
| **Q9** | Sau khi đạt Fmax target, focus tiếp theo là DSP reduction (serialize bn_relu/norm_sq) hay D1 module separation hay D3 const consolidation? | Sau CP-1..3 | Phụ thuộc Q3 (paper scope). Nếu paper kernel-only → D1 ưu tiên. Nếu end-to-end → DSP reduction trước. Trì hoãn quyết định |
| **Q10** | XDC hiện 14ns. Khi nào tighten? Ngay sau CP-1 hay đợi xong CP-3? | Sau CP-1 | **Đề xuất**: sau CP-1 thử 10ns (con đường trung gian) → nếu pass thì sang CP-2/3 với target 8ns. Tighten dần để locate path mới rõ ràng |
| **Q11** | Reset hiện tại async (`posedge rst`). Có giữ async hay đổi sync? | Trước CP-1 (nếu muốn đổi) | Async reset → +0 cycle latency nhưng có timing risk ở reset release. Sync reset → +1 cycle nhưng cleaner CDC. Hệ thống đơn-clock nên không CDC → **giữ async**, không can thiệp |

---

---

## 5. Mapping notes này ↔ FMAX_RESOURCE_PLAN.md

| Item plan | Liên quan note |
|-----------|---------------|
| CP-1 (x_norm_fn pipeline) | G2, đã làm CP-4, sắp làm CP-1 |
| CP-2 (norm_sq pipeline) | G2, O3 |
| CP-3 (S_FIN_WRITE precompute) | G2, G3 (giảm DSP nếu serialize sau) |
| CP-4 (registered strides) | ✅ Done |
| RU-1..RU-4 (RAM layout) | Defer — user yêu cầu skip RAM-related cho tới khi xong Fmax |
| (mới) D1 separation | G1, G5 — chưa có trong plan, cần add |
| (mới) D2 audit SBA | G2 — chưa có trong plan |
| (mới) D3 const consolidation | G3 — chưa có trong plan |
| (mới) D4 BRAM/LUT attribution | G3, cũng chỉ là investigation |

---

## 6. Progress log

### 2026-06-01 — CP-4 done (registered stride incrementers)
Đã xong, không ảnh hưởng Fmax (CP-4 không nằm trên critical path) nhưng giải phóng 4 combinational mults khỏi address path. Free win.

### 2026-06-02 — CP-1 done (RMSNorm_Mul module)
- Module `RMSNorm_Mul.v` mới — 1-stage pipeline giữa 2 mult cascaded.
- M1A_MAC / M1B_MAC: 3 substep → 4 substep.
- xsim TB **PASS** (Final Full Output byte-exact). M6a H_State intermediate fail nhưng downstream (M7, M8, Final) PASS — known acceptable.
- Implementation timing **PASS** @ period 10ns (100 MHz, WNS 0.310ns).
- Worst path vẫn ở `u_rmsnorm_mul/p1_reg` (URAM → 1 DSP → reg, 9.449ns) — không thể giảm thêm trừ khi pipeline URAM output.
- Throughput: 1.27 → **1.66 inf/s** (+30%).
- User confirm 100 MHz đủ, **không tighten Fmax thêm** ở giai đoạn này.

### Tiếp theo (chốt 2026-06-02)
Move sang **resource optimization** theo thứ tự **D3 LUT consolidation → RAM optimization (RU-1..4 + ram_b LUT investigation) → Adder review (O3)**. CP-2/CP-3 defer (chỉ cần nếu sau D3/RAM/Adder vẫn muốn ép Fmax cao hơn).

### 2026-06-02 — D3.A done (architectural cleanup, no resource change)

5 file mới + 1 file sửa:

- **`Silu_LUT.v`** — 256×16 SiLU table, 1 lane, $readmemh init từ `golden_all/silu_lut.txt`
- **`Softplus_LUT.v`** — tương tự softplus
- **`Exp_LUT.v`** — tương tự exp
- **`RSqrt_ROM.v`** — 8K×16 rsqrt ROM, $readmemh từ `golden_all/rsqrt_q97.txt`
- **`Const_Storage.v`** — wrapper duy nhất gộp 48 activation LUT instance (16 lanes × 3 funcs) + 1 RSqrt_ROM. Interface: 6× 256-bit flat ports (silu/sp/exp in/out) + rsqrt idx/data port.
- **`ITM_CONTROLLER.v`** — bỏ generate `Activation_LUT` cũ + bỏ inline `rsqrt_q97_rom` reg → thay bằng 1 instance `Const_Storage`. Pack/unpack array↔flat qua generate `PACK_UNPACK`. Wire `rsqrt_rom_data` thay reference `rsqrt_q97_rom[norm_rom_idx]` (2 chỗ).

**`Activation_LUT.v` (acitivation_lut.v) cũ**: KHÔNG xoá file, chỉ unused. Có thể xoá sau khi confirm TB pass. User sẽ remove khỏi project sources.

**Expected resource impact**: ~0 (cùng underlying primitives, chỉ tái cấu trúc hierarchy). Vivado synthesis có thể hơi khác do hierarchy boundary mới ảnh hưởng physopt — verify bằng re-synth.

### 2026-06-02 (extension) — D3.A full: pull ram_const into Const_Storage too

Theo phản hồi của user, gộp ram_const sang Const_Storage để đúng tinh thần "gộp hết" của advisor.

**Changes**:
- `Const_Storage.v`: thêm `clk` + DMA write port + `const_read_addr/const_read_data` + `dma_rdata_const`. Instantiate `BRAM_256b #(.ADDR_WIDTH(6))` cho ram_const (same as before, just relocated). DMA write triggers khi `dma_target == 2'd3`.
- `Memory_System.v`: bỏ ram_const instance + we_c + addr_b_ram_c + `const_read_addr/const_read_data` ports + dma_rdata branch cho rtarget==3. Comment top-of-file cập nhật DMA target encoding (0/1/2 here, 3 → Const_Storage).
- `ITM_CONTROLLER.v`: thêm wires `mem_dma_rdata`, `const_dma_rdata`. Memory_System dùng `mem_dma_rdata`. Const_Storage giờ nhận DMA + const_read port. Top-level dma_rdata = `(dma_rtarget == 2'd3) ? const_dma_rdata : mem_dma_rdata`.

**Architectural result**:
- Memory_System = R/W working memory only (ram_a/b URAM + ram_w BRAM)
- Const_Storage = TẤT CẢ read-only / config storage (ram_const + activation LUTs + rsqrt ROM)
- DMA target encoding clean: 0/1/2 vào Memory_System, 3 vào Const_Storage

**Resource impact verified (post re-synth)**:
- LUT: 11093 → **10584** (**−509 LUT**, 4.6% reduction — bonus do hierarchy boundary mới giúp Vivado attribute logic gọn hơn)
- BRAM/URAM/DSP unchanged
- WNS 0.336ns @ 10ns — vẫn pass timing
- xsim TB pass (functional byte-exact)

### 2026-06-02 — RAM-2/3/4: Compact map + URAM downsizing

Phân tích lifetime đầy đủ ở `RAM_LAYOUT_PLAN.md`. Triển khai aggressive overlap:

**Bank A layout (peak 17256 words, was ~32000):**
- `[0, 8000)`: A_INPUT_BASE | A_X_INNER | A_BOT_OUT — đều base 0, sequential lifetimes
- `[8000, 16000)`: A_Z_GATE | A_MAMBA_OUT | A_FINAL_OUT — base 8000 / 8000 / 12000
- `[16000, 17000)`: A_CH1_OUT
- `[17000, 17256)`: A_H_STATE

**Bank B layout (peak 19000 words):**
- `[0, 8000)`: B_P1_OUT | B_X_CONV | B_Y_SSM | B_FINAL_OUT — đều base 0, sequential
- `[8000, 16000)`: B_U_SAFE
- `[16000, 19000)`: B_CH2_OUT / B_CH3_OUT / B_CH4_OUT

**BRAM_256b parameterization**: thêm `DEPTH` parameter (default `1 << ADDR_WIDTH`). Memory_System truyền `DEPTH=20480` cho ram_a/ram_b → Vivado infer 5-deep URAM cascade × 4-wide = **20 URAM per bank** thay vì 8×4 = 32.

**Files touched**:
- `ITM_CONTROLLER.v` — localparam block re-mapped
- `BRAM_256b.v` — thêm DEPTH parameter
- `Memory_System.v` — pass DEPTH=20480 cho ram_a/ram_b
- `ITM_CTRL_TB.v` — mirror localparam (TB backward compatible, hierarchical refs vẫn dùng cùng symbol)
- `_block_params.v` — note rằng define cũ là historical only, không reference

**Resource savings VERIFIED (post re-synth + xsim, 2026-06-03)**:
- **URAM**: 64 → **40** (−24 URAM, từ 100% → **62.5%**) ✅ thoát khỏi URAM-bound
- **LUT**: 10584 → **10491** (−93 LUT, cumulative 11093 → 10491 = **−602 LUT / −5.4%**)
- **Register**: 4507 → **4504** (~unchanged)
- **BRAM**: 118 (unchanged, 81.94%)
- **DSP**: 59 (unchanged, 4.73%)
- **WNS**: 0.336 → **0.646 ns** @ 10 ns (100 MHz) — đẹp hơn, thoát critical từ URAM-cascade dài sang path mới
- **WHS**: 0.089 ns (hold pass)
- xsim TB: 5/5 block **Final Full Output PASS** (byte-exact). M6a/intermediate fails giữ nguyên — known FP-rounding artifact, không phải regression.

**Trust model**: user nói TB trust theo Final Full Output. M6a H_STATE intermediate có thể fail (đã fail từ trước CP-1 do FP-rounding khác golden) — không phải bug.

---

## 7. RTL Optimization Phase — **CLOSED 2026-06-03**

Tổng kết hành trình tối ưu từ baseline → final state:

| Metric         | Baseline (pre-CP-1) | Final (2026-06-03) | Δ        |
|----------------|---------------------|--------------------|----------|
| LUT            | 11093               | **10491**          | −602 (−5.4%) |
| Register       | ~4507               | 4504               | ~0       |
| BRAM           | 118                 | 118                | 0        |
| URAM           | 64 (100%)           | **40 (62.5%)**     | **−24 (−37.5%)** |
| DSP            | 59                  | 59                 | 0        |
| WNS @ 10ns     | 0.310 ns (CP-1)     | **0.646 ns**       | +0.336 ns headroom |
| Fmax (achieved)| 100 MHz             | 100 MHz            | (no tighten requested) |
| xsim Final Pass| 5/5 blocks          | 5/5 blocks         | byte-exact giữ nguyên |

**Optimization steps applied**:
1. **CP-1** — `RMSNorm_Mul` pipeline (1-stage giữa 2 cascaded mults) → WNS 0.310 ns @ 10ns
2. **D3.A** — `Const_Storage` consolidation (4 activation/rsqrt + ram_const dồn 1 module) → −509 LUT
3. **RAM-2/3/4** — Compact memory map + `BRAM_256b` `DEPTH` parameterization (5-deep URAM cascade × 4-wide) → −24 URAM, +0.310 ns WNS slack

**Items deferred (paper-scope dependent)**:
- CP-2 (norm_sq pipeline), CP-3 (S_FIN_WRITE precompute) — chưa cần vì 100 MHz đã đạt
- D2 SBA audit — chưa cần vì WNS đã có slack
- O3 adder consolidation — kết luận sơ bộ là không lợi

**Next phase (paper-prep)**: Tách Inception/Mamba làm "explicit version" để extract số liệu riêng cho từng kernel (D1). Xem section 8.

---

## 8. D1 Implementation — Inception/Mamba Explicit Separation

**Mục tiêu**: Có số liệu RTL riêng cho Inception block vs Mamba block, để paper báo cáo fair-comparison với các paper khác (MambaHW chạy riêng, Inception/CNN chạy riêng).

**Scope cần chốt với user trước khi triển khai**: xem AskUserQuestion.

*Last updated: 2026-06-03 — RTL optimization phase closed, transition to paper-prep (D1 separation).*
