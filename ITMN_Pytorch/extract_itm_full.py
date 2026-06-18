"""
extract_itm_full_hwexact.py  —  HW-exact golden extractor for ITMN accelerator.
VERSION: 2.0 FINAL

Every golden is computed with integer arithmetic matching the RTL PE exactly:
  - 40-bit accumulator MAC → >>7 → sat16      (Unified_PE)
  - sat_add16 for bias / residual adds
  - bn_relu function (mul>>7 + shift + relu)
  - Activation LUT emulation (SiLU, softplus, exp) matching Activation_LUT.v
  - SSM scan in integer domain
  - RMSNorm applied inline before M1a/M1b (per-timestep, matches RTL NORM_M1A/M1B substates)
  - Integer chaining across blocks: final_q of block k → input of block k+1 (with hw_maxpool where needed)

Usage:
    python extract_itm_full_hwexact.py --out ./golden_all --all_blocks
    python extract_itm_full_hwexact.py --out ./golden --block_index 0
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

from dataset import get_loaders
from ecg_models.ITMN import ITMN
from utils.utils import get_config

FB = 11
SCALE = 1 << FB  # 2048
RMS_EXTRA_PREC = 6   # N: extra precision bits for RMSNorm ROM index (v2)
                     # Effective target_rms quantum ~0.044 at N=6 (vs old ~0.7 at N=0)

# =====================================================================
#  Low-level int16 helpers
# =====================================================================
def to_f32(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().contiguous().numpy()
    return np.asarray(x, np.float32)

def q(arr):
    """Float → int64 Qfb (floor, clip to int16). Format determined by FB/SCALE."""
    return np.clip(np.floor(np.asarray(arr, np.float64) * SCALE).astype(np.int64),
                   -32768, 32767)

def sat16(x):
    return np.clip(np.asarray(x, np.int64), -32768, 32767)

def sat_add(a, b):
    return sat16(np.asarray(a, np.int64) + np.asarray(b, np.int64))

# =====================================================================
#  File I/O
# =====================================================================
def save_hex(arr_float, path, fb=FB):
    """Save float array as Qfb hex (format determined by fb=FB)."""
    if torch.is_tensor(arr_float): arr_float = to_f32(arr_float)
    a = np.asarray(arr_float, np.float64).ravel()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for v in a:
            iv = max(-32768, min(32767, int(math.floor(float(v) * (1 << fb)))))
            f.write(f'{iv & 0xFFFF:04x}\n')
    print(f'  -> {path.name:<36s} ({a.size} vals)')

def save_iq(arr_int, path):
    """Save int64 Q9.7 array as hex (no requantization)."""
    a = np.asarray(arr_int, np.int64).ravel()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for v in a:
            iv = max(-32768, min(32767, int(v)))
            f.write(f'{iv & 0xFFFF:04x}\n')
    print(f'  -> {path.name:<36s} ({a.size} vals, int-exact)')

def load_hex(path):
    vals = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                v = int(ln, 16)
                if v >= 0x8000: v -= 0x10000
                vals.append(v)
    return np.array(vals, np.int64)

# =====================================================================
#  Activation LUT emulation  (matches Activation_LUT.v)
# =====================================================================
def _build_lut(func_name):
    """256-entry LUT covering float range [-8, +8).
    Index i → x_q = LUT_LO + i * LUT_STEP  where LUT_STEP = SCALE // 16."""
    LUT_STEP = SCALE // 16        # 8 for FB=7, 128 for FB=11
    LUT_LO   = -8 * SCALE         # -1024 for FB=7, -16384 for FB=11
    table = np.zeros(256, np.int64)
    for i in range(256):
        x_q = LUT_LO + i * LUT_STEP
        x_f = x_q / SCALE
        if func_name == 'silu':
            y_f = x_f / (1.0 + math.exp(-x_f))
        elif func_name == 'softplus':
            y_f = math.log1p(math.exp(x_f))
        elif func_name == 'exp':
            y_f = math.exp(x_f)
        else:
            raise ValueError(func_name)
        table[i] = max(-32768, min(32767, int(math.floor(y_f * SCALE))))
    return table

_LUT_SILU = _build_lut('silu')
_LUT_SP   = _build_lut('softplus')
_LUT_EXP  = _build_lut('exp')

# =====================================================================
#  RMSNorm rsqrt ROM  (v2: no pre-shift + finer ROM index, N=RMS_EXTRA_PREC bits)
#  ROM[m] = round(K_new / sqrt(m)),  K_new = sqrt(2^(1+N)) * SCALE.
#  For fb=11, N=6: K_new ≈ 23170, ROM[128] ≈ 2048 = rsqrt(1) Q4.11.
#  m = sum(x^2) >> (log2(d) + 2*fb - 1 - N)  per timestep — no per-channel truncation.
#  Scale factor: y_norm = sat16(sat16((x*gamma)>>fb) * ROM[m] >> fb)
# =====================================================================
def gen_rsqrt_rom(size=8192):
    """v2 ROM: K_new = sqrt(2^(1+N)) * SCALE. Calibrated so that mean_i=2^(1+N)
    when target_rms_float=1.0 (= 128 for N=6). Old K=SCALE^1.5/32 was broken
    because mean_i resolution was only 0.5 unit of target_rms²."""
    K = (2.0 ** ((1 + RMS_EXTRA_PREC) / 2.0)) * float(SCALE)
    rom = [np.int64(32767)]                         # ROM[0]: 1/sqrt(0) = inf → saturate
    for m in range(1, size):
        val = K / math.sqrt(float(m))
        rom.append(np.int64(min(32767, max(-32768, int(round(val))))))
    return rom

_RSQRT_ROM = gen_rsqrt_rom(8192)

def hw_rms_norm_inline(x_q, w_norm_q, d_out, fb=FB):
    """HW-exact RMSNorm v2: no >>5 pre-shift, raw x*x accumulator, single final shift.
    Matches RTL controller's updated norm_sq40_fn + widened norm_sq_acc.
    Fixes (1) per-channel truncation and (2) ROM resolution at small target_rms."""
    T = x_q.shape[1]
    norm_q = np.zeros_like(x_q)
    CH_OUT = d_out // 16
    log2_d = int(round(math.log2(d_out)))
    total_shift = log2_d + 2*fb - 1 - RMS_EXTRA_PREC   # = log2_d + 15 for fb=11, N=6
    for t in range(T):
        norm_sq_acc = np.int64(0)
        for cg in range(CH_OUT):
            word = x_q[cg*16:(cg+1)*16, t].astype(np.int64)
            # v2: accumulate raw x*x (no >>5 pre-shift, no per-channel >>fb)
            # Each lane: |x| ≤ 32767, x² ≤ 2^30. Sum over CH_OUT*16 ≤ 128 lanes: ≤ 2^37.
            norm_sq_acc += int(np.sum(word * word))
        mean_int = int(norm_sq_acc >> total_shift)
        S_t = int(_RSQRT_ROM[min(mean_int, 8191)])
        for ch in range(d_out):
            x_i  = int(x_q[ch, t])
            g_i  = int(w_norm_q[ch])
            p1   = max(-32768, min(32767, (x_i * g_i) >> fb))
            norm_q[ch, t] = max(-32768, min(32767, (p1 * S_t) >> fb))
    return norm_q

def hw_maxpool(x_q, stride=2):
    """Integer max-pool stride=stride over T — matches nn.MaxPool1d(stride, stride)."""
    d, T = x_q.shape
    T_out = T // stride
    out = np.zeros((d, T_out), np.int64)
    for t in range(T_out):
        out[:, t] = np.maximum(x_q[:, t*stride], x_q[:, t*stride + 1])
    return out

def _lut_apply(x_q, table, oor_lo, oor_hi_fn):
    LUT_LO    = -8 * SCALE           # -1024 for FB=7, -16384 for FB=11
    LUT_SHIFT = FB - 4               # 3 for FB=7, 7 for FB=11
    x = np.asarray(x_q, np.int64)
    in_range = (x >= LUT_LO) & (x < -LUT_LO)
    idx = np.where(in_range, ((x - LUT_LO) >> LUT_SHIFT).astype(np.int64), 0)
    idx = np.clip(idx, 0, 255)
    lut_val = table[idx]
    oor_val = np.where(x < LUT_LO, oor_lo, oor_hi_fn(x))
    return np.where(in_range, lut_val, oor_val)

def lut_silu(x_q):
    return _lut_apply(x_q, _LUT_SILU, np.int64(0), lambda x: sat16(x))

def lut_softplus(x_q):
    return _lut_apply(x_q, _LUT_SP, np.int64(0), lambda x: sat16(x))

def lut_exp(x_q):
    return _lut_apply(x_q, _LUT_EXP, np.int64(0), lambda x: np.int64(32767))

# =====================================================================
#  BN helpers  (fold conv+bn, fold standalone bn)
# =====================================================================
def fuse_conv_bn(conv, bn):
    w = to_f32(conv.weight)
    b = to_f32(conv.bias) if conv.bias is not None else np.zeros(w.shape[0])
    g = to_f32(bn.weight); beta = to_f32(bn.bias)
    mu = to_f32(bn.running_mean); var = to_f32(bn.running_var)
    s = g / np.sqrt(var + bn.eps)
    return w * s[:, None, None], b * s + beta - s * mu

def fold_bn(bn):
    g = to_f32(bn.weight); beta = to_f32(bn.bias)
    mu = to_f32(bn.running_mean); var = to_f32(bn.running_var)
    s = g / np.sqrt(var + bn.eps)
    return s, beta - s * mu

# =====================================================================
#  HW-exact PE operations
# =====================================================================
def pe_mac_mv(x_q, w_q, fb=FB):
    """out[co,t] = sat16( (Σ w[co,ci]*x[ci,t]) >> fb )"""
    return sat16((w_q.astype(np.int64) @ x_q.astype(np.int64)) >> fb)

def pe_mac_conv(x_q, w_q, kernel, pad, fb=FB):
    """Full conv1d: accumulate over K taps × C_in channels, then >>fb.
    x_q: (C_in, T)   w_q: (C_out, C_in, K)   → (C_out, T)"""
    C_in, T = x_q.shape
    C_out = w_q.shape[0]
    out = np.zeros((C_out, T), np.int64)
    for t in range(T):
        acc = np.zeros(C_out, np.int64)
        for k in range(kernel):
            t_eff = t + k - pad
            if 0 <= t_eff < T:
                for ci in range(C_in):
                    acc += x_q[ci, t_eff] * w_q[:, ci, k].astype(np.int64)
        out[:, t] = acc
    return sat16(out >> fb)

def pe_mul_vec(a_q, b_q, fb=FB):
    """Element-wise MUL: sat16((a*b) >> fb)."""
    return sat16((a_q.astype(np.int64) * b_q.astype(np.int64)) >> fb)

def bn_relu_hw(raw_q, scale_q, shift_q, fb=FB):
    """Matches RTL bn_relu function exactly."""
    mul = raw_q.astype(np.int64) * scale_q.astype(np.int64)
    bn = sat16(mul >> fb)
    bn = sat16(bn + shift_q.astype(np.int64))
    return np.where(bn < 0, np.int64(0), bn)

# =====================================================================
#  P1:  Conv1D+BN (fused).  k=1 no pad.
# =====================================================================
def hw_p1(x_q, w_q, b_q, fb=FB):
    """out = sat_add( sat16((W @ X) >> fb), bias )"""
    return sat_add(sat16((w_q.astype(np.int64) @ x_q.astype(np.int64)) >> fb),
                   b_q[:, None])

# =====================================================================
#  Hook helpers
# =====================================================================
def is_itm_block(m):
    return hasattr(m, 'conv') and hasattr(m, 'inception_block') and hasattr(m, 'mamba_block')

def register_hooks(model):
    dumps = {}; hooks = []; itm_indices = []
    def mk(name):
        def fn(m, inp, out):
            if name not in dumps: dumps[name] = {}
            dumps[name]['in']  = inp[0].detach().cpu().numpy()[0]
            dumps[name]['out'] = out.detach().cpu().numpy()[0]
        return fn
    for bk, blk in enumerate(model.layers):
        if not is_itm_block(blk): continue
        itm_indices.append(bk)
        hooks.append(blk.conv[0].register_forward_hook(mk(f'P1_Conv_{bk}')))
        hooks.append(blk.conv[1].register_forward_hook(mk(f'P1_BN_{bk}')))
        hooks.append(blk.mamba_block.register_forward_hook(mk(f'MambaBlock_{bk}')))
        hooks.append(blk.register_forward_hook(mk(f'ITMBlock_{bk}')))
    return dumps, hooks, itm_indices

# =====================================================================
#  MAMBA — full integer pipeline
# =====================================================================
def extract_mamba_hwexact(mamba_block, p1_q, out_dir, fb=FB):
    """
    Full Mamba in HW-exact integer domain.
    RMSNorm is applied inline before M1a/M1b (matching RTL NORM_M1A/M1B substates).
    p1_norm = hw_rms_norm_inline(p1_q, norm.weight)
    M1a = W_x @ p1_norm, M1b = W_z @ p1_norm.
    """
    mixer = mamba_block.mixer
    d_inner = mixer.in_proj.weight.shape[0] // 2
    d_state = mixer.A_log.shape[1]
    dt_rank = mixer.dt_proj.in_features
    d_in    = mixer.in_proj.weight.shape[1]
    T       = p1_q.shape[1]

    print(f'  [MAMBA HW] d_in={d_in} d_inner={d_inner} d_state={d_state} '
          f'dt_rank={dt_rank} T={T}')

    # Quantise weights
    w_mx_q  = q(to_f32(mixer.in_proj.weight[:d_inner, :]))
    w_mz_q  = q(to_f32(mixer.in_proj.weight[d_inner:, :]))
    w_dw_q  = q(to_f32(mixer.conv1d.weight)).reshape(d_inner, 4)
    b_dw_q  = q(to_f32(mixer.conv1d.bias))

    n_act   = dt_rank + 2 * d_state
    n_pad   = ((n_act + 15) // 16) * 16
    xpw     = to_f32(mixer.x_proj.weight)
    xp_padded = np.zeros((n_pad, xpw.shape[1]), np.float32)
    xp_padded[:n_act] = xpw
    w_xp_q  = q(xp_padded)

    w_dt_q  = q(to_f32(mixer.dt_proj.weight))
    b_dt_q  = q(to_f32(mixer.dt_proj.bias))
    A_q     = q(to_f32(-torch.exp(mixer.A_log)))
    D_q     = q(to_f32(mixer.D))
    w_out_q = q(to_f32(mixer.out_proj.weight))

    # ---- RMSNorm (inline, matches RTL NORM_M1A/M1B states) ----
    w_norm_q = q(to_f32(mamba_block.norm.weight))
    p1_norm_q = hw_rms_norm_inline(p1_q, w_norm_q, d_in, fb=fb)
    save_iq(p1_norm_q, out_dir / 'P1_Norm_Output_FP.txt')

    # ---- M1a: x_inner = in_proj_x(p1_norm) ----
    x_inner_q = pe_mac_mv(p1_norm_q, w_mx_q)
    save_iq(x_inner_q, out_dir / 'Mam_X_Inner_FP.txt')

    # ---- M1b: z_gate = in_proj_z(p1_norm) ----
    z_gate_q = pe_mac_mv(p1_norm_q, w_mz_q)
    save_iq(z_gate_q, out_dir / 'Mam_Z_Gate_FP.txt')

    # ---- M2: depthwise conv1d k=4 causal pad=3 + bias ----
    x_conv_q = np.zeros((d_inner, T), np.int64)
    for d in range(d_inner):
        for t in range(T):
            acc = np.int64(0)
            for k in range(4):
                t_eff = t + k - 3
                if 0 <= t_eff < T:
                    acc += x_inner_q[d, t_eff] * w_dw_q[d, k]
            x_conv_q[d, t] = acc
    x_conv_q = sat_add(sat16(x_conv_q >> fb), b_dw_q[:, None])
    save_iq(x_conv_q, out_dir / 'Mam_X_Conv_FP.txt')

    # ---- M3: SiLU ----
    u_q = lut_silu(x_conv_q)
    save_iq(u_q, out_dir / 'Mam_U_Silu_FP.txt')

    # ---- M4: x_proj  d_inner → n_pad ----
    xproj_q = pe_mac_mv(u_q, w_xp_q)
    save_iq(xproj_q, out_dir / 'Mam_X_Proj_FP.txt')

    # ---- M5: dt_proj + bias + softplus ----
    dt_raw_q = xproj_q[:dt_rank, :]
    delta_q = np.zeros((d_inner, T), np.int64)
    for t in range(T):
        for g in range(d_inner // 16):
            acc = np.zeros(16, np.int64)
            for r in range(dt_rank):
                acc += dt_raw_q[r, t] * w_dt_q[g*16:(g+1)*16, r].astype(np.int64)
            delta_q[g*16:(g+1)*16, t] = sat_add(sat16(acc >> fb),
                                                  b_dt_q[g*16:(g+1)*16])
    delta_q = lut_softplus(delta_q)
    save_iq(delta_q, out_dir / 'Mam_Delta_FP.txt')

    # ---- M6: SSM scan ----
    B_q = xproj_q[dt_rank:dt_rank+d_state, :]
    C_q = xproj_q[dt_rank+d_state:dt_rank+2*d_state, :]
    h_q = np.zeros((d_inner, d_state), np.int64)
    y_ssm_q = np.zeros((d_inner, T), np.int64)

    for t in range(T):
        for s in range(d_state):
            da_in = pe_mul_vec(delta_q[:, t], A_q[:, s])
            dA    = lut_exp(da_in)
            dB    = pe_mul_vec(delta_q[:, t], np.full(d_inner, B_q[s, t], np.int64))
            t1    = pe_mul_vec(dA, h_q[:, s])
            t2    = pe_mul_vec(dB, u_q[:, t])
            h_q[:, s] = sat_add(t1, t2)

        for g in range(d_inner // 16):
            acc = np.zeros(16, np.int64)
            for s in range(d_state):
                acc += C_q[s, t] * h_q[g*16:(g+1)*16, s].astype(np.int64)
            y_ch = sat16(acc >> fb)
            du   = pe_mul_vec(D_q[g*16:(g+1)*16], u_q[g*16:(g+1)*16, t])
            y_ssm_q[g*16:(g+1)*16, t] = sat_add(y_ch, du)

        if t % 200 == 0:
            print(f'    SSM t={t}/{T}', end='\r')
    print(f'    SSM done.                ')
    save_iq(h_q, out_dir / 'Mam_H_State_FP.txt')

    # ---- M7: y_gated = y_ssm * SiLU(z) ----
    y_gated_q = pe_mul_vec(y_ssm_q, lut_silu(z_gate_q))
    save_iq(y_gated_q, out_dir / 'Mam_Y_Gated_FP.txt')

    # ---- M8: out_proj  d_inner → d_in ----
    mamba_out_q = pe_mac_mv(y_gated_q, w_out_q)
    save_iq(mamba_out_q, out_dir / 'Mam_OutProj_FP.txt')

    return mamba_out_q, h_q

# =====================================================================
#  Per-block extraction
# =====================================================================
def extract_block(model, bk, out_dir, dumps, x_q_chain=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    blk = model.layers[bk]
    print(f"\n{'='*60}\n  EXTRACT BLOCK [{bk}] → {out_dir}\n{'='*60}")

    d_in  = blk.conv[0].in_channels
    d_out = blk.conv[0].out_channels

    # ================================================================
    # PHASE 1 — Conv1D + BN (fused)
    # ================================================================
    print('\n[1] P1: Conv1D+BN (HW-exact)')
    w_fused, b_fused = fuse_conv_bn(blk.conv[0], blk.conv[1])
    save_hex(w_fused, out_dir / 'P1_Weight_Fused.txt')
    save_hex(b_fused, out_dir / 'P1_Bias_Fused.txt')

    if x_q_chain is not None:
        x_q = x_q_chain                 # integer-domain chain from previous block
        T   = x_q.shape[1]
        save_iq(x_q, out_dir / 'P1_Input_X.txt')
    else:
        p1_in_float = dumps[f'P1_Conv_{bk}']['in']
        T   = p1_in_float.shape[-1]
        x_q = q(p1_in_float)
        save_hex(p1_in_float, out_dir / 'P1_Input_X.txt')

    print(f'  d_in={d_in}  d_out={d_out}  T={T}')

    w_q = q(w_fused.reshape(d_out, d_in))
    b_q = q(b_fused)

    p1_q = hw_p1(x_q, w_q, b_q)      # (d_out, T)
    save_iq(p1_q, out_dir / 'P1_Output_Golden_FP.txt')

    # ================================================================
    # PHASE 2 — Inception (HW-exact)
    # ================================================================
    print('\n[2] Inception (HW-exact)')
    inc = blk.inception_block
    dim = d_out // 4   # channels per branch (16 for blk 0-3, 32 for blk 4)

    # Save weights
    save_hex(inc.bottleneck.weight, out_dir / 'W_Bot.txt')
    save_hex(inc.conv1.weight,      out_dir / 'W_B1.txt')
    save_hex(inc.conv2.weight,      out_dir / 'W_B2.txt')
    save_hex(inc.conv3.weight,      out_dir / 'W_B3.txt')
    save_hex(inc.conv4.weight,      out_dir / 'W_B4.txt')
    scale_inc, shift_inc = fold_bn(inc.bn)
    save_hex(scale_inc, out_dir / 'Inc_BN_Scale.txt')
    save_hex(shift_inc, out_dir / 'Inc_BN_Shift.txt')

    # Bot: (dim, d_out) k=1 no-bias, input = P1_out
    w_bot_q = q(to_f32(inc.bottleneck.weight).reshape(dim, d_out))
    bot_q   = pe_mac_mv(p1_q, w_bot_q)
    save_iq(bot_q, out_dir / 'Out_Bot_FP.txt')
    save_iq(bot_q, out_dir / 'Out_Bot.txt')

    # B1: (dim, d_out) k=1 no-bias, input = maxpool(P1_out)
    # MaxPool1d(kernel=3, stride=1, pad=1) — same length, different values
    p1_mp_q = np.zeros_like(p1_q)
    for t in range(T):
        t0 = max(0, t - 1); t2 = min(T - 1, t + 1)
        p1_mp_q[:, t] = np.maximum(np.maximum(p1_q[:, t0], p1_q[:, t]), p1_q[:, t2])
    w_b1_q = q(to_f32(inc.conv1.weight).reshape(dim, d_out))
    b1_q   = pe_mac_mv(p1_mp_q, w_b1_q)
    save_iq(b1_q, out_dir / 'Out_B1_FP.txt')
    save_iq(b1_q, out_dir / 'Out_B1.txt')

    # B2: (dim, dim, 9) conv k=9 pad=4 on Bot_out
    w_b2_np = to_f32(inc.conv2.weight)
    w_b2_q  = q(w_b2_np.reshape(dim, dim, w_b2_np.shape[2]))
    b2_q    = pe_mac_conv(bot_q, w_b2_q, kernel=9, pad=4)
    save_iq(b2_q, out_dir / 'Out_B2_FP.txt')
    save_iq(b2_q, out_dir / 'Out_B2.txt')

    # B3: conv k=19 pad=9 on Bot_out
    w_b3_np = to_f32(inc.conv3.weight)
    w_b3_q  = q(w_b3_np.reshape(dim, dim, w_b3_np.shape[2]))
    b3_q    = pe_mac_conv(bot_q, w_b3_q, kernel=19, pad=9)
    save_iq(b3_q, out_dir / 'Out_B3_FP.txt')
    save_iq(b3_q, out_dir / 'Out_B3.txt')

    # B4: conv k=39 pad=19 on Bot_out
    w_b4_np = to_f32(inc.conv4.weight)
    w_b4_q  = q(w_b4_np.reshape(dim, dim, w_b4_np.shape[2]))
    b4_q    = pe_mac_conv(bot_q, w_b4_q, kernel=39, pad=19)
    save_iq(b4_q, out_dir / 'Out_B4_FP.txt')
    save_iq(b4_q, out_dir / 'Out_B4.txt')

    # Concatenate: (B1, B2, B3, B4) each (dim, T) → (d_out, T)
    inc_cat_q = np.concatenate([b1_q, b2_q, b3_q, b4_q], axis=0)

    # ================================================================
    # PHASE 3 — Mamba (HW-exact, RMSNorm inline before M1a/M1b)
    # ================================================================
    print('\n[3] Mamba (HW-exact)')
    mixer = blk.mamba_block.mixer

    # Save weights (same format as TB expects)
    save_hex(blk.mamba_block.norm.weight, out_dir / 'Mam_W_Norm.txt')
    inproj_w = to_f32(mixer.in_proj.weight)
    d_inner  = inproj_w.shape[0] // 2
    save_hex(inproj_w[:d_inner, :], out_dir / 'Mam_W_InProj_X.txt')
    save_hex(inproj_w[d_inner:, :], out_dir / 'Mam_W_InProj_Z.txt')
    save_hex(mixer.conv1d.weight,   out_dir / 'Mam_W_Conv.txt')
    save_hex(mixer.conv1d.bias,     out_dir / 'Mam_B_Conv.txt')

    dt_rank  = mixer.dt_proj.in_features
    d_state  = mixer.A_log.shape[1]
    n_act    = dt_rank + 2 * d_state
    n_pad    = ((n_act + 15) // 16) * 16
    xpw      = to_f32(mixer.x_proj.weight)
    xp_padded = np.zeros((n_pad, xpw.shape[1]), np.float32)
    xp_padded[:n_act] = xpw
    save_hex(xp_padded, out_dir / 'Mam_W_XProj.txt')
    save_hex(xp_padded, out_dir / 'Mam_W_xProj.txt')
    save_hex(mixer.dt_proj.weight, out_dir / 'Mam_W_DtProj.txt')
    save_hex(mixer.dt_proj.weight, out_dir / 'Mam_W_dtProj.txt')
    save_hex(mixer.dt_proj.bias,   out_dir / 'Mam_B_DtProj.txt')
    save_hex(mixer.dt_proj.bias,   out_dir / 'Mam_B_dtProj.txt')
    save_hex(-torch.exp(mixer.A_log), out_dir / 'Mam_A_signed.txt')
    save_hex(mixer.D,              out_dir / 'Mam_D_param.txt')
    save_hex(mixer.out_proj.weight, out_dir / 'Mam_W_OutProj.txt')

    mamba_out_q, h_q = extract_mamba_hwexact(blk.mamba_block, p1_q, out_dir)

    # ================================================================
    # PHASE 4 — Final: relu(bn(inc_cat)) + relu(mamba_out)  (PyTorch formula)
    # ================================================================
    print('\n[4] Final (PyTorch formula: relu(bn(inc_cat)) + relu(mamba_out))')
    # Matches ITMN.py ITMBlock.forward exactly:
    #   x1 = inception_block(x)              -> relu(bn(inc_cat))
    #   x2 = relu(mamba_block(x))            -> relu(mamba_out)
    #   out = x1 + x2
    # The HW workaround formula bn_relu(inc+mam) loses ~0.37 AUC at FB=11 vs float;
    # this PyTorch-faithful formula closes the gap to <0.003 AUC (test_hw fb11_py).
    scale_q = q(scale_inc)
    shift_q = q(shift_inc)

    final_q = np.zeros((d_out, T), np.int64)
    for ch in range(d_out):
        x1 = bn_relu_hw(inc_cat_q[ch, :],
                        np.full(T, scale_q[ch], np.int64),
                        np.full(T, shift_q[ch], np.int64))
        x2 = np.where(mamba_out_q[ch, :] < 0, np.int64(0), mamba_out_q[ch, :])
        final_q[ch, :] = sat_add(x1, x2)
    save_iq(final_q, out_dir / 'Final_ITM_Full_FP.txt')

    # Float-reference P1 for backward compat
    save_hex(dumps[f'P1_BN_{bk}']['out'], out_dir / 'P1_Output_Golden.txt')

    print(f'\n  Block {bk} extraction complete.')
    return dict(p1_q=p1_q, inc_cat_q=inc_cat_q,
                mamba_out_q=mamba_out_q, final_q=final_q)

# =====================================================================
#  LUT hex files for RTL $readmemh
# =====================================================================
def save_lut_files(out_dir):
    out_dir = Path(out_dir)
    for name, table in [('silu_lut.txt', _LUT_SILU),
                        ('softplus_lut.txt', _LUT_SP),
                        ('exp_lut.txt', _LUT_EXP)]:
        p = out_dir / name
        with open(p, 'w') as f:
            for v in table:
                f.write(f'{int(v) & 0xFFFF:04x}\n')
        print(f'  -> {name:<36s} (256 entries)')
    # rsqrt Q9.7 ROM for RMSNorm
    p = out_dir / 'rsqrt_q97.txt'
    with open(p, 'w') as f:
        for v in _RSQRT_ROM:
            f.write(f'{int(v) & 0xFFFF:04x}\n')
    print(f'  -> rsqrt_q97.txt                     (8192 entries)')

# =====================================================================
#  Accuracy analysis helpers
# =====================================================================
def block_sqnr(float_ref, int_q, scale=SCALE):
    """SQNR (dB) between PyTorch float output and HW integer output."""
    hw = int_q.astype(np.float64) / scale
    sig  = np.mean(float_ref.astype(np.float64) ** 2)
    noise = np.mean((hw - float_ref.astype(np.float64)) ** 2)
    return 10.0 * np.log10(sig / noise) if noise > 0 and sig > 0 else (np.inf if noise == 0 else -np.inf)

def analyze_accuracy(model, dumps, itm_indices, block_results, dev):
    """
    Per-block SQNR comparison and single-sample classifier prediction check.
    Also prints Mamba-only SQNR to isolate integer SSM drift from formula error.
    """
    print(f"\n{'='*60}")
    print('  ACCURACY ANALYSIS  (single sample)')
    print(f"{'='*60}")

    # --- Mamba-only SQNR diagnostic ---
    print(f"\n  [Mamba SQNR — integer vs float, before final add]")
    print(f"  {'Block':<8} {'Mamba_shape':<20} {'SQNR (dB)':<12} {'Max int16 sat%'}")
    print(f"  {'-'*60}")
    for idx, bk in enumerate(itm_indices):
        mam_q   = block_results[idx]['mamba_out_q']        # (d_out, T) int64
        # Hook captures MambaBlock output before transpose back → shape (T, d_out)
        mam_f   = dumps[f'MambaBlock_{bk}']['out'].T      # → (d_out, T) float32
        sqnr    = block_sqnr(mam_f, mam_q)
        sat_pct = 100.0 * np.mean(np.abs(mam_q) >= 32767)
        print(f"  bk={bk} (blk{idx}) {str(mam_f.shape):<20} {sqnr:<12.2f} {sat_pct:.1f}%")
    print()

    print(f"  {'Block':<8} {'PyTorch_out shape':<22} {'SQNR (dB)':<12} {'RMS err':<12} {'Max |err|'}")
    print(f"  {'-'*70}")

    last_hw_float = None
    for idx, bk in enumerate(itm_indices):
        final_q   = block_results[idx]['final_q']          # (d_out, T) integer
        pt_out    = dumps[f'ITMBlock_{bk}']['out']         # (d_out, T) float32

        sqnr  = block_sqnr(pt_out, final_q)
        hw    = final_q.astype(np.float64) / SCALE
        rms   = np.sqrt(np.mean((hw - pt_out.astype(np.float64)) ** 2))
        mx    = np.max(np.abs(hw - pt_out.astype(np.float64)))

        label = f'bk={bk} (blk{idx})'
        print(f"  {label:<8} {str(pt_out.shape):<22} {sqnr:<12.2f} {rms:<12.5f} {mx:.5f}")

        last_hw_float = torch.tensor(hw.astype(np.float32)).unsqueeze(0)  # (1,d_out,T)

    # Classifier comparison (last block output → GAP → linear)
    print(f"\n  Classifier prediction (integer chain vs float model):")
    # Float model prediction
    pt_last = torch.tensor(dumps[f'ITMBlock_{itm_indices[-1]}']['out']).unsqueeze(0).to(dev)
    with torch.no_grad():
        pt_mean   = pt_last.mean(dim=-1)
        pt_logits = model.classifier(pt_mean)
        pt_pred   = pt_logits.argmax(dim=-1).item()
        pt_conf   = torch.softmax(pt_logits, dim=-1).max().item()

    # HW integer chain prediction
    hw_last = last_hw_float.to(dev)
    with torch.no_grad():
        hw_mean   = hw_last.mean(dim=-1)
        hw_logits = model.classifier(hw_mean)
        hw_pred   = hw_logits.argmax(dim=-1).item()
        hw_conf   = torch.softmax(hw_logits, dim=-1).max().item()

    match = 'MATCH' if hw_pred == pt_pred else 'MISMATCH'
    print(f"  Float  model: class={pt_pred}  conf={pt_conf:.3f}")
    print(f"  HW int chain: class={hw_pred}  conf={hw_conf:.3f}  [{match}]")
    print(f"{'='*60}\n")

# =====================================================================
#  MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_type', default='super')
    ap.add_argument('--out', default='./golden_all')
    ap.add_argument('--block_index', type=int, default=0)
    ap.add_argument('--all_blocks', action='store_true')
    ap.add_argument('--lut_dir', default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    lut_dir = Path(args.lut_dir) if args.lut_dir else out_root
    print('[LUT] Generating activation LUT hex files...')
    save_lut_files(lut_dir)

    print('\n[MODEL] Loading...')
    params = get_config('config.yaml')
    params['exp_type'] = args.exp_type.lower()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader, num_class, _ = get_loaders(
        params['data'], params['exp_type'], params['hyperparameters']['batch_size'])
    model = ITMN(n_classes=num_class, **params['model']).to(dev)
    model.load_state_dict(
        torch.load(params['test_ckpt_path'], map_location=dev)['model_state_dict'])
    model.eval()

    sample = next(iter(test_loader))
    if isinstance(sample, dict):
        for key in ['waveform', 'signal', 'x', 'input']:
            if key in sample:
                waveform = sample[key][:1].to(dev); break
        else:
            raise ValueError(f'No input key: {list(sample.keys())}')
    elif isinstance(sample, (list, tuple)):
        waveform = sample[0][:1].to(dev)
    else:
        waveform = sample[:1].to(dev)

    print('[HOOKS] Registering...')
    dumps, hooks, itm_indices = register_hooks(model)
    print(f'  ITM blocks at indices: {itm_indices}')

    print('[FWD] Running...')
    with torch.no_grad():
        _ = model(waveform)
    for h in hooks: h.remove()
    print(f'  Captured {len(dumps)} hook outputs')

    if args.all_blocks:
        prev_final_q = None
        prev_bk      = None
        block_results = []
        for idx, bk in enumerate(itm_indices):
            blk_dir = out_root / f'block_{idx:02d}_layer{bk:02d}'
            x_q_for_block = None
            if prev_final_q is not None:
                has_maxpool = any(
                    isinstance(model.layers[j], torch.nn.MaxPool1d)
                    for j in range(prev_bk + 1, bk)
                )
                x_q_for_block = hw_maxpool(prev_final_q) if has_maxpool else prev_final_q
            result = extract_block(model, bk, blk_dir, dumps, x_q_chain=x_q_for_block)
            block_results.append(result)
            prev_final_q = result['final_q']
            prev_bk      = bk
        print(f"\n{'='*60}\n  ALL {len(itm_indices)} BLOCKS DONE → {out_root}\n{'='*60}")
        analyze_accuracy(model, dumps, itm_indices, block_results, dev)
    else:
        bk = args.block_index
        if bk not in itm_indices:
            print(f'[ERROR] layers[{bk}] not ITM. Available: {itm_indices}')
            return
        extract_block(model, bk, out_root, dumps)

if __name__ == '__main__':
    main()