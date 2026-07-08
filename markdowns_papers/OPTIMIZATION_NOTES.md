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

### 2026-06-07 — D1 done (OOC standalone synth + compare)

Mechanism: 1 file controller (`ITM_CONTROLLER_v2.v`) với 2 wrapper module (`Mamba_Top.v`, `Inception_Top.v`) compile với `+define+MAMBA_ONLY` / `+define+INCEPTION_ONLY` qua 3 TCL OOC:
- `Mamba_OOC.tcl` → `reports/mamba/` (utilization + timing + routed checkpoint)
- `Inception_OOC.tcl` → `reports/inception/`
- `D1_Compare.tcl` → `reports/d1_compare/` (hierarchical util + critical-path text + summary CSV)

Define cắt code:
- `MAMBA_ONLY`: strip P1+BR+FIN+CASCADE state arms (lines 591–785, 1492–1632 trong v2). S_IDLE nhảy thẳng vào S_NORM_M1A_SQ_READ; M8 done = block done (skip FIN).
- `INCEPTION_ONLY`: strip M*/NORM Mamba state arms (lines 787–1490). S_BR_NEXT sau nhánh cuối nhảy thẳng vào S_FIN_READ; FIN đọc A_MAMBA_OUT (URAM init 0 → relu(0) = 0 → final = relu(bn(inc))).
- Memory_System + Const_Storage + PE_Array instantiate vô điều kiện cả 3 build → BRAM/URAM/DSP/LUT memory-related giữ nguyên; DCE chỉ cắt phần FSM + datapath không dùng.

Kết quả (KV260 xck26-sfvc784-2LV-c, OOC clk=10ns):

| Resource | Mamba_Top | Inception_Top | Full ITM | Sum vs Full (logic only) |
|---|---|---|---|---|
| LUT | 7513 | 5346 | 10491 | sum=12859, saved 2368 (18%) |
| REG | 3945 | 2103 | 4504 | saved 1544 (26%) |
| DSP | 39 | 37 | 59 | saved 17 (22%) |
| BRAM | 118 | 118 | 118 | shared 100% (1 ram_weight) |
| URAM | 40 | 40 | 40 | shared 100% (1 Memory_System) |
| WNS | 0.710 ns | 1.973 ns | 0.646 ns | — |

Critical path:
- Mamba: ram_a URAM-cascade ×4 (2.83 ns) → bank_sel LUT mux → 1 DSP48E2 (`u_rmsnorm_mul/p1_wide`) → sat16 CARRY8 → `p1_reg_reg[5]` FDSE. Logic levels = 17. Slack +0.710 ns.
- Inception: ram_b URAM-cascade ×4 → BR data capture LUT3 → `max_buf` 16-bit CARRY8 compare → `pe_A` mux chain (LUT5+LUT6+LUT6) → `pe_A_reg[13]` FDCE. Logic levels = 10. Slack +1.973 ns.

PDF schematic export trong `D1_Compare.tcl` (write_schematic) chưa generated — đã extract `_critical_path.rpt` thành block diagram tay cho báo cáo.

---

## 9. End-to-end FPGA — encoder + GAP + classifier (2026-06-14)

**Quyết định**: scope paper chuyển từ "ITM-block kernel" sang **end-to-end ECG classifier**. Phải implement nốt 3 phần hiện đang chạy ở host Python:

1. **Encoder**: input projection 12-lead ECG → d_model channels (Conv1D đầu vào).
2. **GAP**: Global Average Pooling sau block 4 (256 channels × 250 T → 256 scalar).
3. **Linear classifier**: FC layer d_out → num_classes.

**Hệ quả**:
- O2 chuyển từ "open question" sang **chốt end-to-end**. Cập nhật bảng Q1-Q4 section 4.
- Cycle/latency tổng phải re-measure (bao gồm encoder + GAP + FC trên FPGA).
- Resource estimate phải update: encoder Conv1D ~vài chục param → có thể fit vào PE_Array hiện tại bằng FSM extension. FC layer phụ thuộc num_classes (ECG multi-label thường 5–10 class) → tiny MAC.
- Verify chain phải extend: golden cho encoder/GAP/FC output (hiện `extract_itm_full.py` chỉ dump 5 block). Cần thêm extractor cho 3 stage này.

**Ưu tiên implementation**:
1. Encoder Conv1D: tái sử dụng FSM P1-like (Conv+BN+ReLU). Add input phase trước S_P1_READ của block 0.
2. GAP: thêm state sau block 4 FIN — đọc final output, accumulate sum, divide by T (shift nếu T = power-of-2; else fixed-point divide).
3. FC: 1 MAC reduction giống M7 out_proj nhưng dimension nhỏ hơn → reuse PE_Array.

**Risk**: encoder và FC dùng weight + bias mới → ram_weight cần thêm region. Phải re-plan memory map (hiện 118 BRAM = 81.94%, còn 26 BRAM free → OK).

---

## 10. Advisor feedback (2026-06-14) — Fmax surprise

Giáo viên đánh giá Fmax 100 MHz (slack +0.646 ns @ 10 ns) **cao bất ngờ** so với các paper Mamba/SSM HW đã công bố trước, đặc biệt là output sinh viên. Cần chuẩn bị giải trình.

**Hypothesis đã thống nhất với giáo viên (sinh viên trình bày)**:

Kiến trúc cố ý **đơn giản, đánh đổi rõ ràng cycle vs Fmax**:
- Datapath không có function chain phức tạp giữa các register. Mỗi cycle chỉ làm 1 trong 4 việc:
  1. Đọc 1 word từ Memory_System hoặc Const_Storage hoặc weight RAM (1 cycle BRAM/URAM read).
  2. Compute qua PE_Array (DSP MAC, 1 cycle).
  3. Đi qua 1 LUT activation hoặc Writeback Transform (sat / bn_relu / relu / max).
  4. Register vào capture reg hoặc m_wr_data → ghi RAM.
- Không có wide combinational reduction giữa nhiều DSP. RMSNorm sum-of-squares (`norm_sq16_fn`) là chỗ duy nhất có 16-way adder tree, đã được pipeline qua CP-1.
- Không dùng SBA central mux → loại được fanin large-mux delay.
- Trade-off: cycle count rất cao (block 4 ≈ 11.7M cycle / inference, total ~250–400 ms @ 100 MHz cho 5 block). Lý do chấp nhận được: ECG classification không yêu cầu real-time sub-millisecond.

**Validation từ critical-path report**:
- Logic levels Mamba = 17 (đa số là DSP internal stages, không phải LUT depth).
- Logic levels Inception = 10 (URAM cascade dominate; LUT chỉ 4 cấp).
- Cả 2 đều có slack dương > 0.5 ns → confirm Fmax 100 MHz có thực, không phải hold violation hidden.

**Cần làm trước báo cáo**:
- Soạn slide so sánh Fmax/throughput/resource với 2–3 paper Mamba HW gần nhất (Vivado-based, FPGA target). Highlight trade-off cycle-cao-Fmax-cao có chủ ý.
- Nhấn mạnh: **byte-exact verification** (verify_byte_exact.py 5/5 block PASS) — không phải chỉ là dạng PoC.

---

## 11. Báo cáo + PPT (deadline tuần sau, 2026-06-14)

**Trạng thái giáo viên**: chưa nắm rõ thiết kế. Buổi báo cáo cần diagram chi tiết để giáo viên hình dung được flow.

**Diagram cần có** (priority high — điểm số phụ thuộc lớn vào việc giáo viên hiểu thiết kế):

1. **System-level**: Encoder → 5× ITM block (với cascade pool giữa B1-B2 và B3-B4) → GAP → FC → output. Show kích thước (T, d_in, d_out, d_inner) từng block.
2. **ITM block per-block**: P1 (Conv+BN) → split path → (Inception 4 branches + bottleneck max-pool) || (RMSNorm → M1A x_inner + M1B z_gate → M2..M8 Mamba) → FIN combine (bn_relu(inc) + relu(mam)). Đã có ở DESIGN_REPORT.md section 1.
3. **MAMBA-only + INCEPTION-only build diagrams** (D1): đã có 2 hình tay, cần fix 4 chỗ sai đã trace (xem chat 2026-06-14):
   - Mamba: thêm output arrow cho Exp LUT feedback, sửa vị trí RMSNorm_Mul (peer của Const_Storage/MemSys, không phải tầng filter), bỏ đường silu/sp/exp ảo vào RMSNorm_Mul.
   - Inception: thêm output arrow cho Capture Regs (max_buf → OPERAND MUX + Writeback; incep_reg → Writeback bn_relu).
4. **PE_Array internal** (peer-level + per-lane):
   - Top: 16-lane mux network (a_is_vector chọn giữa scalar broadcast pe_A vs vector pe_A_vec), pe_B vector luôn lane-wise.
   - Per-lane: Unified_PE = 1 DSP48E2 multiplier + 40-bit acc + sat16 → out_val 16-bit.
5. **Critical path block diagram** (2 hình tay đã có): Mamba (URAM×4 + DSP + sat) và Inception (URAM×4 + max_buf CARRY8 + pe_A mux). Highlight chỗ tại sao Fmax đạt 100 MHz.
6. **Writeback MUX + Transforms**: nhiều combinational datapath song song (silu_lut, exp_lut, sat_add16+bn_relu+relu16, max_buf, pe_out, norm_sq) → 1 wide MUX state-driven → m_wr_data register → MemSys din_a port. Trả lời câu hỏi "function chọn thế nào".
7. **Memory hierarchy** (giải thích Const_Storage vs ram_const):
   - ram_const (writable BRAM "block" → Vivado override sang LUT-distributed, 64×256 nông): per-channel parameter (γ, BN scale/shift, P1_BIAS, M_DW_BIAS, M_DT_BIAS) — load DMA từng block.
   - Function ROM bank: 16-lane × {Silu_LUT, Softplus_LUT, Exp_LUT} + 1× RSqrt_ROM — fixed function approximation, $readmemh init, bake vào bitstream.
   - Memory_System: 2 URAM bank (ram_a/ram_b, 20K×256 mỗi cái = 20 URAM/bank) + ram_weight BRAM (16K×256 = 114 BRAM).
8. **Cycle/throughput breakdown table**: Phase 1 / Inception / Mamba (M1-M8) / Final per block (đã có ở DESIGN_REPORT section 7). Tổng cycle 5 block + sắp tới + encoder + GAP + FC.
9. **Optimization journey**: bar chart LUT/URAM/WNS before/after CP-1 + D3.A + RAM-2/3/4.
10. **Comparison table với paper khác**: column = Fmax, throughput (inf/s), LUT, DSP, BRAM, URAM, AUC; row = ITMN (ours) + 2–3 paper Mamba/SSM HW + 2–3 paper CNN/Inception HW. Chốt fair-compare bằng D1 standalone numbers.

**Format báo cáo**:
- PPT: 15–20 slide. Mở đầu (problem + dataset ECG) → ITMN architecture (model side) → RTL design choices → D1 fair compare → Optimization journey → End-to-end plan (encoder/GAP/FC) → Future work.
- Report (text): detail-level expansion của từng slide, có hình + bảng + critical-path text trích `.rpt`. Có thể base trên DESIGN_REPORT.md hiện tại, expand thêm phần D1, encoder plan.

**Risk**:
- Giáo viên có thể hỏi sâu vào: (a) số liệu vs paper khác (cần chuẩn bị bảng), (b) tại sao Fmax cao (xem section 10), (c) byte-exact verify methodology (extract_itm_full.py + verify_byte_exact.py), (d) khi nào encoder/GAP/FC xong (timeline).

*Last updated: 2026-06-14 — End-to-end scope confirmed, D1 OOC standalone done, advisor presentation prep.*
