# ITMN Mamba RTL — Design Notes

## TODO: Refactor FSM sang per-timestep pipeline

**Quyết định:** Sau khi tất cả standalone stage TBs pass (M1A/B, M2, M3, M4, M5, M6, M7, M8, RMSNorm), refactor `Mamba_Top.v` FSM từ batch mode sang per-timestep pipeline.

---

### Vấn đề với batch mode hiện tại

FSM hiện tại xử lý **toàn bộ T timestep** cho từng stage trước khi sang stage tiếp:

```
[M1A × T=1000] → store x_inner[T] → [M1B × T] → store z_gate[T] → ...
```

Điều này yêu cầu RAM lớn để lưu tạm các tensor trung gian:
- x_inner, z_gate, x_conv, u, delta, y_ssm, y_gated: mỗi tensor 8000 words (128ch × T=1000)
- Peak address = 16000 + 999×8 + 7 = **23999** → cần DEPTH=24576 (24 URAM/bank)

---

### Thiết kế per-timestep (mục tiêu)

Outer loop theo t, inner loop chạy toàn bộ stages cho mỗi t:

```
for t = 0 to T-1:
    M1A(t): x_inner = W_x × x_norm[t]
    M1B(t): z_gate  = W_z × x_norm[t]
    M2(t):  x_conv  = DepthConv(x_inner[t-3..t]) + b_dw   ← 4-tap causal buffer
    M3(t):  u       = SiLU(x_conv)
    M4(t):  x_proj  = W_xproj × u
    M5(t):  delta   = softplus(W_dt × x_proj[0:4] + b_dt)
    M6(t):  y_ssm   = SSM_scan(delta, u, B, C, h)          ← h persistent across t
    M7(t):  y_gated = y_ssm × SiLU(z_gate)
    M8(t):  out[t]  = W_out × y_gated
```

### RAM requirement sau refactor

| Buffer | Size | Words (256-bit) |
|--------|------|-----------------|
| x_inner circular buf (4 tap M2) | 4 × d_inner = 512 values | 32 |
| z_gate (hold M1B→M7) | d_inner = 128 values | 8 |
| u (hold M3→M4/M6) | d_inner = 128 values | 8 |
| x_proj (hold M4→M5/M6) | n_pad = 48 values | 3 |
| delta (hold M5→M6) | d_inner = 128 values | 8 |
| y_ssm (hold M6→M7) | d_inner = 128 values | 8 |
| H_RegFile (SSM state) | d_inner × d_state = 2048 values | 128 (đã có) |
| **Tổng ram_a + ram_b** | | **~100–150 words** |

So với batch: **24000 words → ~150 words, tiết kiệm ~160×**.

Synth: từ 24 URAM/bank → 1–2 BRAM nhỏ.

---

### Điểm cần xử lý khi refactor

1. **M2 circular buffer**: x_inner không còn lưu full T. Thay bằng circular buffer 4-deep (32 words). FSM dùng `t mod 4` làm index tap.

2. **z_gate**: phải giữ cho đến M7 (7 stages sau). Dùng 1 register word (8 words) trong ram hoặc register file.

3. **M1A/M1B**: MAC loop vẫn giống hệt, chỉ không lặp over T ở đây — T loop chuyển lên outer FSM.

4. **M4/M8 (MAC over input dim)**: tương tự M1A, chỉ đổi dimension. Không thay đổi inner MAC.

5. **M6**: Đã per-timestep, không cần đổi gì.

6. **State machine**: Thêm outer state `S_T_LOOP` / `S_T_NEXT`. Xóa inner T-loop trong từng stage.

7. **TBs standalone**: Giữ nguyên để regression test. Sau refactor thêm `tb_Mamba_Top_FULL.v` test full pipeline T=1000.

---

### Không ảnh hưởng

- Weights (ram_weight): không thay đổi layout, vẫn load 1 lần trước khi start.
- H_RegFile: không thay đổi, đã thiết kế cho per-timestep.
- Standalone TBs (M1A–M8, RMSNorm): vẫn chạy được vì test từng stage độc lập.
- DEPTH=24576 trong Memory_System.v: giữ nguyên cho batch standalone tests, sẽ giảm sau refactor.

---

*Ghi chú: 2026-07-01. Refactor sau khi tất cả standalone TBs pass.*
