# RAM Layout & Lifetime Plan

Phân tích memory map hiện tại, lifetime mỗi region, và đề xuất overlap để giảm URAM/BRAM usage.

## 1. Memory map hiện tại

### Bank A (ram_a, URAM, 32K × 256-bit)
```
Addr        Region          Lifetime (phase write → last read)
─────────────────────────────────────────────────────────────────
0           A_INPUT_BASE    P1 read; CASCADE write (next block)
4000        A_BOT_OUT       BR sub-step write → BR sub-step read (1 phase only!)
5000        A_CH1_OUT       BR ch1 write → FIN read
8000        A_FINAL_OUT     FIN c_grp≥1 write → host/CASCADE read
12000       A_X_INNER       M1A write → M2 read; M3 write (u); M5 write (delta) → M6A read
20000       A_Z_GATE        M1B write → M7 read
28000       A_H_STATE       M6A_INIT write → M6B last read (256 words for block 4)
28128       A_MAMBA_OUT     M8 write → FIN read  ← OVERLAPS A_H_STATE for block 4!
─────────────────────────────────────────────────────────────────
End used    ≈ 28256 (block 4) / ≈ 32128 (block 0)
```

### Bank B (ram_b, URAM, 32K × 256-bit)
```
Addr        Region          Lifetime
─────────────────────────────────────────────────────────────────
0           B_P1_OUT        P1 write → M1A/M1B/M2 read
4000        B_CH2_OUT       BR ch2 write → FIN read
5000        B_CH3_OUT       BR ch3 write → FIN read
6000        B_CH4_OUT       BR ch4 write → FIN read
8000        B_FINAL_OUT     FIN c_grp=0 write → host/CASCADE read
12000       B_X_CONV        M2 write (x_conv) → M3 read; M4 write (x_proj) → M5 read
15000       B_U_SAFE        M3CP write → M6A_T2/M6B_DU read
23000       B_Y_SSM         M6B write → M7 read
─────────────────────────────────────────────────────────────────
End used    ≈ 31000 (block 0) / ≈ 27000 (block 4)
```

---

## 2. Lifetime matrix — region × FSM phase

Marker: `W` = write, `R` = read, `─` = idle, `D` = dead (region no longer needed).

| Region (bank A) | P1 | BR | M1A | M1B | M2 | M3 | M3CP | M4 | M5 | M6A | M6B | M7 | M8 | FIN | CASC |
|----------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| A_INPUT_BASE   | R  | ─  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | ─   | W    |
| A_BOT_OUT      | ─  | W+R| **D**| D| D  | D  | D    | D  | D  | D   | D   | D  | D  | D   | D    |
| A_CH1_OUT      | ─  | W  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | R   | D    |
| A_FINAL_OUT    | ─  | ─  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | W   | R    |
| A_X_INNER      | ─  | ─  | W  | ─  | R  | W (u) | R  | R  | W (delta) | R | ─ | ─  | ─  | D   | D    |
| A_Z_GATE       | ─  | ─  | ─  | W  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | R  | D  | D   | D    |
| A_H_STATE      | ─  | ─  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | W+R | W+R | D  | D  | D   | D    |
| A_MAMBA_OUT    | ─  | ─  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | W  | R   | D    |

| Region (bank B) | P1 | BR | M1A | M1B | M2 | M3 | M3CP | M4 | M5 | M6A | M6B | M7 | M8 | FIN | CASC |
|----------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| B_P1_OUT       | W  | R  | R  | R  | R  | D  | D    | D  | D  | D   | D   | D  | D  | D   | D    |
| B_CH2_OUT      | ─  | W  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | R   | D    |
| B_CH3_OUT      | ─  | W  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | R   | D    |
| B_CH4_OUT      | ─  | W  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | R   | D    |
| B_FINAL_OUT    | ─  | ─  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | ─   | ─  | ─  | W   | R    |
| B_X_CONV       | ─  | ─  | ─  | ─  | W  | R  | ─    | W (xp)| R | ─  | ─   | ─  | ─  | D   | D    |
| B_U_SAFE       | ─  | ─  | ─  | ─  | ─  | ─  | W    | ─  | ─  | R   | R   | D  | D  | D   | D    |
| B_Y_SSM        | ─  | ─  | ─  | ─  | ─  | ─  | ─    | ─  | ─  | ─   | W   | R  | D  | D   | D    |

---

## 3. Word counts per region per block

br_dim_groups = `(CH_OUT == 8) ? 2 : 1`. dim = `CH_OUT * 4`. d_inner = CH_M × 16.

| Region            | Formula                        | Blocks 0,1 (T=1000, CH_OUT=4, CH_M=8) | Block 4 (T=250, CH_OUT=8, CH_M=16) |
|------------------|--------------------------------|-----|------|
| A_INPUT_BASE      | T × CH_IN                       | 4000 | 1000 |
| B_P1_OUT          | T × CH_OUT                      | 4000 | 2000 |
| A_BOT_OUT         | T × CH_OUT                      | 4000 | 2000 |
| A_CH1_OUT         | T × br_dim_groups               | 1000 | 500  |
| B_CH2/3/4_OUT     | T × br_dim_groups (mỗi region) | 1000 | 500  |
| A_X_INNER         | T × CH_M                        | 8000 | 4000 |
| A_Z_GATE          | T × CH_M                        | 8000 | 4000 |
| B_X_CONV (M2)     | T × CH_M                        | 8000 | 4000 |
| B_X_CONV (M4)     | T × 3 (x_proj)                  | 3000 | 750  |
| B_U_SAFE          | T × CH_M                        | 8000 | 4000 |
| A_H_STATE         | CH_M × 16 (= d_state)           | 128  | 256  |
| A_MAMBA_OUT       | T × CH_OUT                      | 4000 | 2000 |
| B_Y_SSM           | T × CH_M                        | 8000 | 4000 |
| A_FINAL_OUT       | T × CH_OUT (c_grp≥1 slots)      | 4000 | 2000 |
| B_FINAL_OUT       | T (c_grp=0 only)                | 1000 | 250  |

**Aggregate per bank, no overlap (block 0)**:
- Bank A: 4000+4000+1000+4000+8000+8000+128+4000 = **33,128 words** (vượt 32K!)
- Bank B: 4000+1000×3+1000+8000+8000+8000 = **32,000 words** (sát limit)

Hiện tại "không overlap" sẽ vượt URAM. Các overlap đã có:
- A_BOT_OUT/A_CH1_OUT chia sẻ [4000..8000) range
- A_H_STATE và A_MAMBA_OUT chồng [28000..28256)
- A_FINAL_OUT [8000..12000) chia sẻ với A_X_INNER nếu c_grp=0 (đi vào B)

---

## 4. Overwrite opportunities (impact-ranked)

### **OW-1**: `A_BOT_OUT` (4000 words) — dead sau BR, có thể overlap với A_H_STATE/A_MAMBA_OUT

**Hiện tại**: A_BOT_OUT [4000, 8000) dead từ M1A trở đi. Nhưng A_H_STATE và A_MAMBA_OUT đang ở [28000, 32000). Có thể di chuyển 2 region này về [4000, 8000):

```
A_H_STATE   = 4000   (size 256, ends at 4256)
A_MAMBA_OUT = 4256   (size 4000, ends at 8256)
```

**Lợi**: Free vùng [28000, 32000) = 4000 words trên ram_a → có thể giảm ram_a depth từ 32K xuống ≤16K.

**Cẩn thận**: A_FINAL_OUT đang ở 8000. A_MAMBA_OUT mới = 4256..8256 sẽ chạm A_FINAL_OUT. Cần xác minh A_MAMBA_OUT đã dead trước khi A_FINAL_OUT bắt đầu write. → Đúng: M8 ghi MAMBA_OUT, FIN đọc MAMBA_OUT + ghi FINAL_OUT. FIN đọc XONG MAMBA_OUT trước khi ghi FINAL_OUT (per c_grp loop). Nên có thể overlap nhưng RISKY — A_MAMBA_OUT đọc nhiều lần trong FIN, từng c_grp. Tránh đặt cùng vị trí.

→ **An toàn nếu** A_FINAL_OUT (8000) > A_MAMBA_OUT_end (8256). KHÔNG OK với layout trên — A_MAMBA_OUT ends at 8256 > A_FINAL_OUT base 8000. Conflict.

**Sửa**: Đặt A_MAMBA_OUT vào [4256, 8256) NHƯNG A_FINAL_OUT cũng phải dời. A_FINAL_OUT chỉ dead sau host đọc → có thể đặt sau M_X_INNER nếu X_INNER dead trước FIN.

Wait — A_X_INNER (12000..20000) dead sau M6A_DA reading delta. M6A_T2 reads B_U_SAFE (not X_INNER), so X_INNER dead from end of M6A onward. FIN runs after M8. So A_X_INNER region [12000, 20000) **dead during FIN** → can host A_FINAL_OUT.

```
Proposed:
A_H_STATE    = 4000   (256 words → [4000, 4256))
A_MAMBA_OUT  = 4256   (4000 words → [4256, 8256))   block 0 max
A_FINAL_OUT  = 12000  (reuse A_X_INNER region)      block 0: 4000 words → [12000, 16000)
```

**Net gain**: free [8000, 12000) (4000 words trên ram_a) + free [28000, 32000) (4000 words).

→ Bank A peak usage: [0, 20000) ≈ 20K words. URAM depth có thể giảm xuống **20K** thay vì 32K.

---

### **OW-2**: `A_Z_GATE` (8000 words) — dead sau M7

A_Z_GATE [20000, 28000) used M1B..M7. After M7, dead.

**Có thể overlap với gì?** A_MAMBA_OUT (M8 write → FIN read) — nhưng M8 đến sau M7. Lifecycle disjoint ✓

→ Di chuyển A_MAMBA_OUT vào A_Z_GATE region: `A_MAMBA_OUT = 20000`.

Kết hợp với OW-1: ngay từ đầu loại bỏ vùng [28000, 32000) hoàn toàn.

```
Refined proposal:
A_H_STATE    = 4000   (256 words → [4000, 4256))   reuse A_BOT_OUT region
A_Z_GATE     = 12000  reuse A_X_INNER tail after X_INNER dead
                       BUT A_Z_GATE written M1B, dead by M7, while X_INNER
                       written by M1A → DIFFERENT timing. M1A writes X_INNER
                       at [12000, 20000); M1B writes Z_GATE — needs another region.
```

→ A_Z_GATE conflict-free placement: must avoid A_X_INNER's lifetime. Hiện tại OK ở [20000, 28000).

**Pragmatic**: A_MAMBA_OUT (4000 words) → put at [20000, 24000) reuse Z_GATE region first half (Z_GATE dead by M7, MAMBA_OUT alive from M8).

```
Proposed bank A layout:
0      A_INPUT_BASE                 4000 words
4000   A_H_STATE | A_BOT_OUT        4000 words (temporal overlap)
8000   A_CH1_OUT | A_FINAL_OUT      4000 words (temporal overlap, both write at separate phases)
12000  A_X_INNER                    8000 words
20000  A_Z_GATE | A_MAMBA_OUT       8000 words (Z dead by M7, MAMBA written M8)
─────────────────────────────────
Total peak: 28000 words → URAM 32K → could use 28K or stay 32K
```

**Save**: ~4000 words = 4 URAM tiles potentially saved if depth reduced to 28K. But URAM is 4K-deep per primitive, so saving must be in 4K increments. 32K → 28K saves 4 URAM in depth dimension × 4 wide = 0 (still need same wide stack for 256-bit width). Actually 28K = 7×4K so 7-deep × 4-wide = 28 URAM per bank. Currently 8×4 = 32 URAM per bank. **Save 8 URAM total (2 banks)**.

---

### **OW-3**: `B_U_SAFE` (8000 words) — dead sau M6B

**Có thể overlap với B_Y_SSM?** B_U_SAFE [15000, 23000) read by M6B's last cycles. B_Y_SSM written DURING M6B (per t_cnt loop) → could conflict per-cycle.

Cụ thể: trong M6B loop, mỗi t reads U_SAFE (du) và writes Y_SSM. Nếu cùng region → conflict.

→ KHÔNG safe to overlap U_SAFE và Y_SSM.

**Nhưng B_U_SAFE có thể overlap với B_CH2/3/4_OUT?** CH2-4 dead during M-phases (only read in FIN). U_SAFE alive M3CP..M6B → overlap ranges [M3CP..M6B] vs [dead] — disjoint! ✓

→ Có thể đặt B_U_SAFE ở [4000, 12000) (where CH2-4 currently live)? Nhưng CH2-4 phải được preserve cho FIN read sau M-phases.

→ Tóm: NẾU M3CP eliminated (RU-1), thì B_U_SAFE không tồn tại nữa → free 8000 words.

---

### **OW-4 / RU-1**: Eliminate M3CP entirely

Currently M3 → M3CP → M4 → M5. M3CP just copies A_X_INNER (u) to B_U_SAFE.

**Lý do M3CP tồn tại**: M5 ghi delta lên A_X_INNER, đè u. Nên cần copy u đi đâu đó.

**Cách bỏ M3CP**:
- Đổi M3 write target: M3 ghi u trực tiếp vào **B_U_SAFE** (thay vì A_X_INNER).
- M4 đọc u từ B_U_SAFE (thay vì A_X_INNER).
- M5 ghi delta vào A_X_INNER (như cũ).
- M6A_DA đọc delta từ A_X_INNER (như cũ).
- M6A_T2, M6B_DU đọc u từ B_U_SAFE (như cũ).

**Cycle saving**: M3CP có 5 substep × CH_M × T cycles per block. 5×8×1000 = 40K cycles per block (0-3). ~5 blocks → **~140K cycles total = 0.25% inference**. Modest.

**Code change**: medium — sửa M3_WRITE target, M4_READ source. Sạch hơn về flow.

---

### **OW-5**: Ping-pong write opportunity — currently mỗi state set bank_sel explicitly

Bank-sel chuyển thường xuyên qua các state để route read/write giữa ram_a/ram_b. Cách clean hơn:
- Định rõ "owner bank" cho từng region.
- Bank_sel switching minimal — chỉ khi cross-bank dataflow.

Sẽ phân tích kỹ ở Step 2 (sau khi confirm overwrite plan).

---

## 5. Đề xuất layout mới (combined)

### Bank A (target: peak 24K thay vì 32K)
```
Addr   Region                Lifetime
─────────────────────────────────────────
0      A_INPUT_BASE          P1 read, CASCADE write
4000   A_H_STATE+A_BOT_OUT   BR write → M6 use → die       (overlap, temporal disjoint)
8000   A_CH1_OUT             BR write → FIN read
12000  A_X_INNER             M1A..M6A (multi-content)
20000  A_Z_GATE+A_MAMBA_OUT  M1B..M7 (Z), M8..FIN (MAMBA)  (overlap, temporal disjoint)
24000  A_FINAL_OUT           FIN write → CASCADE/host read
─────────────────────────────────────────
Peak: 28000 (if A_FINAL_OUT 4000 size for block 0)
URAM depth: 28K → 7-deep cascade × 4-wide = 28 URAM per bank (vs 32 hiện tại)
```

### Bank B (target: peak 24K thay vì 32K)
```
Addr   Region                Lifetime
─────────────────────────────────────────
0      B_P1_OUT              P1..M2 read
4000   B_CH2_OUT             BR..FIN
5000   B_CH3_OUT             BR..FIN
6000   B_CH4_OUT             BR..FIN
8000   B_FINAL_OUT           FIN..end
12000  B_X_CONV              M2..M5 (multi-content)
20000  B_U_SAFE              M3..M6B (or remove via RU-1)
                              ↓
       Nếu RU-1: free B_U_SAFE entirely, B_Y_SSM shifts to [20000, 28000)
─────────────────────────────────────────
Peak nếu giữ U_SAFE: 28K. Nếu RU-1: 28K (B_Y_SSM moves to 20K, ends 28K).
```

---

## 6. Implementation order

| Step | Change | Lines touched | Risk |
|------|--------|--------------|------|
| **6.1** | OW-1: Move A_H_STATE 28000 → 4000 (reuse BOT_OUT region) | 2 localparams | Low — temporal disjoint verified |
| **6.2** | OW-2: Move A_MAMBA_OUT 28128 → 20000 (reuse Z_GATE second half) | 1 localparam | Low — disjoint M7 vs M8 |
| **6.3** | OW-1b: Move A_FINAL_OUT 8000 → 24000 (after free) | 1 localparam | Medium — TB hierarchical refs need update |
| **6.4** | RU-1: Eliminate M3CP (M3 → B_U_SAFE, M4 reads from B) | Several state machine changes | Medium-high — careful FSM rework |
| **6.5** | Reduce URAM depth: BRAM_256b ADDR_WIDTH 15 → 14 (16K) hoặc giữ 15 (no save) | Memory_System | Low after compact map |

→ Bắt đầu từ **6.1, 6.2, 6.3** (chỉ đổi localparam, không đụng FSM) → re-run TB verify byte-exact → re-synth check URAM count.

Sau khi confirm thì sang 6.4 (M3CP elimination) và 6.5 (declaration size).

---

## 7. Post-mortem: Bug discovered in v1 attempt (2026-06-02)

**Triệu chứng**: Block 0 Final 38% fail, Block 4 Final 73% fail. Block 0 Inception PASS (B1-B4), Block 4 Inception FAIL.

**Nguyên nhân root cause**: ban đầu tôi đặt `B_Y_SSM = 0` với rationale "P1_OUT (P1..M1B) → X_CONV (M2..M5) → Y_SSM (M6B..M7) → FINAL_OUT" sequential.

**Sai ở đâu**: x_proj data trong B_X_CONV không chỉ sống tới M5. Còn được đọc:
- `S_M6A_DB_READ` (line 1159): đọc B_X_CONV cho B scalar mỗi (t, s)
- `S_M6B_RC_READ` (line 1289): đọc B_X_CONV cho C scalar mỗi (t, s)

→ x_proj alive M4..**M6B**, không phải M2..M5. Trong M6 loop:
- M6A đọc X_CONV @ [0, T*3) cho B scalar
- M6B ghi Y_SSM @ [0, T*ch_m) (sharing với X_CONV)
- M6B's Y_SSM write at [0, 8) overwrites X_CONV's x_proj at [0, T*3)
- M6A next-t read at [t*3, t*3+offset) → đọc Y_SSM data thay vì x_proj → toàn bộ M6/M7/M8/FIN sai

**Fix**: chuyển `B_Y_SSM` từ `0` sang `8000`. Chia sẻ với `B_U_SAFE @ 8000` thay vì với X_CONV. Single-pass overlap U_SAFE↔Y_SSM trong M6B an toàn vì:
- Mỗi (t, c_grp_m) iter: read u tại addr X → write y_ssm cùng addr X
- Iter sau dùng addr X+1 (chưa bị ghi)
- M6A đọc B_X_CONV ở Bank B [0, T*3) — không bị ghi bởi M6B nữa ✓

**Phòng ngừa**: khi lifetime analysis, phải audit TẤT CẢ reads/writes per region, không chỉ dựa vào tên phase. Use `grep` để liệt kê thực sự.

---

*Last updated: 2026-06-02 sau khi fix B_Y_SSM bug.*
