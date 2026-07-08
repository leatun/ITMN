# Testbenches — sources_v3

## tb_Mamba_PE.v

Unit test thuần combinational/sequential cho 1 instance `Mamba_PE`. Test 6 case:
IDLE, MUL, ADD, MAC chain 3 sample, SSM fused, sat boundary. Không cần golden file.

Run (Vivado batch):
```bash
xvlog -i ../common -i ../mamba ../common/_parameter.v ../mamba/Mamba_PE.v tb_Mamba_PE.v
xelab -debug typical tb_Mamba_PE -s tb_pe
xsim tb_pe -R
```

Hoặc trong Vivado GUI: add sources `_parameter.v`, `Mamba_PE.v`, set sim top = `tb_Mamba_PE`, Run All.

Pass khi cuối log có `===== TB DONE — all checks passed (errors=0) =====`.

---

## tb_M_Cluster_P1.v — byte-exact với golden

Drive `M_Cluster` qua P1 stage (MAC + ADD bias), compare từng output với
golden `P1_Output_Golden_FP.txt`. Pass = chứng minh datapath MAC + ADD đúng
arithmetic với reference Python (cùng pattern dùng cho M1A/M1B/M3/M4/M8).

### Chuẩn bị golden

Copy 4 file từ `ITMN_Pytorch/golden_all/block_00_layer00/` vào sim working dir:

```bash
SIMDIR=<vivado_sim_dir>   # nơi xsim chạy, vd: itmn_synth.sim/sim_1/behav/xsim
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Input_X.txt          $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Weight_Fused.txt     $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Bias_Fused.txt       $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Output_Golden_FP.txt $SIMDIR/
```

### Run

```bash
xvlog -i ../common -i ../mamba \
      ../common/_parameter.v \
      ../mamba/Mamba_PE.v ../mamba/H_RegFile.v ../mamba/Reduce16.v ../mamba/M_Cluster.v \
      tb_M_Cluster_P1.v
xelab -debug typical tb_M_Cluster_P1 -s tb_p1
xsim tb_p1 -R
```

Hoặc Vivado GUI: add 5 file source + tb, set sim top = `tb_M_Cluster_P1`,
copy 4 golden file vào sim launch directory, Run All.

### Pass criteria

Cuối log:
```
---- tb_M_Cluster_P1 summary ----
  timesteps tested : 2 / 1000
  total compares   : 128
  errors           : 0
===== TB P1 BYTE-EXACT PASS =====
```

128 compares = 2 timestep × 64 output channel (4 group × 16 lane).
Bật `T_TEST=1000` trong TB để sweep toàn bộ (64000 compares, ~2s sim).

### Nếu fail

Output dạng:
```
FAIL  t=0 c_out=5  got=1234 (0x04D2)  exp=1230 (0x04CE)
```

→ kiểm tra:
1. File golden đúng version (regenerate qua `itmn_pipeline.py extract --out ./golden_all --all_blocks` nếu nghi ngờ).
2. PE arithmetic — `Mamba_PE.v` shift `>>> FRAC_BITS`, sat16 boundary.
3. Bias add: ADD mode Mamba_PE = sat16(W1+H), match `sat_add` Python.

---

## tb_Mamba_Top_RMSNorm.v — byte-exact (RMSNorm stage)

Drive Mamba_Top với `run_stage=4'd9`. Load P1_Output input vào ram_b, gamma
vào ram_const, run, read X_NORM from ram_a, compare.

### Chuẩn bị golden

```bash
SIMDIR=<vivado_sim_dir>
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_W_Norm.txt          $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Output_Golden_FP.txt $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Norm_Output_FP.txt   $SIMDIR/  # ← regen nếu thiếu
```

### Pass criteria

```
---- tb_Mamba_Top_RMSNorm summary ----
  timesteps tested : 4 / 1000
  total compares   : 256
  errors           : 0
===== TB RMSNorm BYTE-EXACT PASS =====
```

---

## tb_Mamba_Top_M1AB.v — byte-exact (M1A + M1B stages)

Drive `Mamba_Top` 2 lần: lần 1 với `run_stage=0` (M1A), lần 2 với `run_stage=1`
(M1B). Cùng input `X_NORM = P1_Norm_Output_FP`, weight khác nhau.

### Chuẩn bị golden

```bash
SIMDIR=<vivado_sim_dir>
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_W_InProj_X.txt    $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_W_InProj_Z.txt    $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/P1_Norm_Output_FP.txt $SIMDIR/  # ← cần regen nếu chưa có
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_X_Inner_FP.txt    $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_Z_Gate_FP.txt     $SIMDIR/
```

**Lưu ý**: `P1_Norm_Output_FP.txt` không có trong project's `golden_all/` (extract
cũ). Cần re-run `itmn_pipeline.py extract --out ./golden_all --all_blocks` trong WSL
hoặc copy từ WSL `/home/letun/ITMN_Latest/golden_all/`.

### Pass criteria

```
---- tb_Mamba_Top_M1AB summary ----
  timesteps tested : 4 / 1000
  M1A compares     : 512   errors: 0
  M1B compares     : 512   errors: 0
===== TB M1A+M1B BYTE-EXACT PASS =====
```

512 compares per stage = 4 timestep × 128 c_out (8 group × 16 lane).

---

## tb_Mamba_Top_M8.v — byte-exact (M8 stage)

Drive `Mamba_Top` qua DMA load + FSM run + DMA readback. Test stage M8
(out_proj) standalone. Pass = chứng minh full pipeline: Memory_System +
M_Cluster + FSM (PREFETCH/WAIT/MAC/FINAL/WRITE/NEXT) hoạt động đúng.

### Chuẩn bị golden

```bash
SIMDIR=<vivado_sim_dir>
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_W_OutProj.txt   $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_Y_Gated_FP.txt  $SIMDIR/
cp ITMN_Pytorch/golden_all/block_00_layer00/Mam_OutProj_FP.txt  $SIMDIR/
```

### Run

```bash
xvlog -i ../common -i ../mamba -i ../top \
      ../common/_parameter.v \
      ../common/BRAM_256b.v ../common/Memory_System.v ../common/Const_Storage.v \
      ../common/Silu_LUT.v ../common/Softplus_LUT.v ../common/Exp_LUT.v ../common/RSqrt_ROM.v \
      ../mamba/Mamba_PE.v ../mamba/H_RegFile.v ../mamba/Reduce16.v ../mamba/M_Cluster.v \
      ../top/Mamba_Top.v \
      tb_Mamba_Top_M8.v
xelab -debug typical tb_Mamba_Top_M8 -s tb_m8
xsim tb_m8 -R
```

### Pass criteria

```
---- tb_Mamba_Top_M8 summary ----
  timesteps tested : 4 / 1000
  total compares   : 256
  errors           : 0
===== TB M8 BYTE-EXACT PASS =====
```

256 compares = 4 timestep × 64 c_out (4 group × 16 lane). Bật `T_TEST=1000`
để sweep toàn bộ block 0.

### Sim time estimate

T_TEST=4: ~10K cycle ≈ 100 μs. T_TEST=1000: ~540K cycle ≈ 5.4 ms.

---

## TB còn thiếu (sẽ viết theo phase A7b/c/d)

- **tb_Mamba_Top_M1A_M1B** — stage M1A + M1B linear, cùng FSM pattern như M8.
  Cần golden `Mam_X_Inner_FP`, `Mam_Z_Gate_FP`, `Mam_W_InProj_X/Z`,
  `P1_Norm_Output_FP`.
- **tb_Mamba_Top_RMSNorm_M2** — RMSNorm + depthwise conv + SiLU.
- **tb_Mamba_Top_M6_SSM** — verify mode SSM. Cần script Python sinh dA/dB
  intermediate vì `itmn_pipeline.py` không dump 2 tensor này.
- **tb_Mamba_Top_full** — full Mamba pipeline end-to-end.
