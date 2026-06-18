"""
test_hw.py — Full HW integer chain accuracy evaluation on test set.

Pipeline per sample:
  integer encoder (hw_p1) → 5×infer_block_hw → GAP → float classifier → sigmoid
Reports AUC/TPR vs float baseline.

Variants (--variant flag):
  hw           Default: Q9.7 integer chain, integer encoder  (RTL-exact)
  ideal_q97    Float PyTorch last-block output → Q9.7 → classifier (AUC ceiling, free)
  float_enc    Float encoder output → Q9.7 → integer block chain
  float_mamba  Integer inception + float Mamba branch (tests Mamba quantisation impact)
  fb11         Q4.11 integer chain (wider fractional bits; nonlinears approximated)
  fb15         Q1.15 integer chain (maximum fractional bits; nonlinears approximated)
  all          Run all variants above

hw / float_enc / float_mamba: integer ops match extract_itm_full.py exactly.
fb11 / fb15: LUT nonlinears replaced by float approximation (diagnostic only, not RTL-exact).
"""
import argparse, math
import numpy as np
import torch
import tqdm

from extract_itm_full import (
    q, to_f32, sat16, sat_add, SCALE, FB,
    fuse_conv_bn, fold_bn,
    hw_p1, pe_mac_mv, pe_mul_vec,
    hw_maxpool,
    lut_silu, lut_softplus, lut_exp,
    is_itm_block, _RSQRT_ROM, RMS_EXTRA_PREC,
    _LUT_SILU, _LUT_SP, _LUT_EXP,
)
from dataset import get_loaders
from ecg_models.ITMN import ITMN
from utils.utils import get_config
from utils.metrics import calculate_metrics

# Convert ROM list → numpy array once for O(1) vectorised lookup
_RSQRT_ROM_NP = np.array(_RSQRT_ROM, np.int64)


# ─────────────────────────────────────────────────────────────────────
#  RMSNorm v2 — no pre-shift + finer ROM index
#
#  Old (v1) bugs:
#    Bug 1: per-channel (x_sh*x_sh) >> fb truncates small x → sum biased
#    Bug 2: ROM K=2896 → mean_i resolution = 0.5 unit of target_rms² →
#           target_rms < ~0.7 all map to mean_i=0 → ROM[0]=32767 saturates →
#           output amplified ~16× (very wrong)
#
#  v2 fixes both with combined approach:
#    1) Accumulate raw x*x in 64-bit (no per-channel truncation)
#    2) Use finer ROM (K_new = sqrt(2^(1+N)) * SCALE) with N extra precision bits
#       → mean_i = M0 when target_rms=1.0  where  M0 = 2^(1+N) = 128 for N=6
#       → target_rms quantum ~ 0.044 for N=6 (vs old ~0.7)
#
#  Total shift = log2_d + 2*FB - 1 - N   (= log2_d + 15 for fb=11, N=6)
# ─────────────────────────────────────────────────────────────────────

def gen_rsqrt_rom_v2(size=8192, fb=FB, extra_prec=RMS_EXTRA_PREC):
    """ROM_V2[m] = round(K_new / sqrt(m))  where K_new = sqrt(2^(1+N)) * 2^fb.
    For fb=11, N=6: K_new ≈ 23170. ROM[0] saturates to 32767.
    Local copy kept for backward-compat fb11_rmsv2 variant; identical to imported _RSQRT_ROM."""
    K_new = (2.0 ** ((1 + extra_prec) / 2.0)) * float(1 << fb)
    rom = [np.int64(32767)]
    for m in range(1, size):
        val = K_new / math.sqrt(float(m))
        rom.append(np.int64(min(32767, max(-32768, int(round(val))))))
    return rom

_RSQRT_ROM_V2_NP = np.asarray(gen_rsqrt_rom_v2(extra_prec=RMS_EXTRA_PREC), np.int64)


def rmsnorm_hw_v2(x_q, w_norm_q, d_out, fb=FB, extra_prec=RMS_EXTRA_PREC):
    """RMSNorm v2 — no pre-shift, full 64-bit accumulator, finer ROM index.
    After Phase 2 port, this is functionally identical to rmsnorm_hw_fast (same v2 algorithm).
    Kept for fb11_rmsv2 variant backward-compat."""
    log2_d      = int(round(math.log2(d_out)))
    x           = x_q.astype(np.int64)
    sq          = x * x                                    # (d_out, T) — full 32-bit per element
    sum_d       = sq.sum(axis=0)                           # (T,) int64 — up to ~2^37 for d=128
    total_shift = log2_d + 2*fb - 1 - extra_prec           # = log2_d + 15 for fb=11, N=6
    mean_i      = sum_d >> total_shift
    S_t         = _RSQRT_ROM_V2_NP[np.clip(mean_i, 0, 8191)]
    p1          = sat16((x * w_norm_q[:, None].astype(np.int64)) >> fb)
    return sat16((p1.astype(np.int64) * S_t[None, :]) >> fb)


def q_fb(arr, fb):
    """Quantize float → int64 fixed-point with fb fractional bits."""
    s = 1 << fb
    return np.clip(np.floor(np.asarray(arr, np.float64) * s).astype(np.int64),
                   -32768, 32767)


# ─────────────────────────────────────────────────────────────────────
#  Vectorised helpers  (int64-exact, same result as nested loops)
# ─────────────────────────────────────────────────────────────────────

def pe_mac_conv_fast(x_q, w_q, kernel, pad, fb=FB):
    """Conv1d via einsum — identical to pe_mac_conv for int64 inputs.
    Max int64 pre-shift: C_in × K × 32767² ≤ 16×39×32767² ≈ 6.7×10¹¹  (safe).
    Einsum 'ctk,ock->ot' output order is (o=C_out, t=T) — no transpose needed."""
    C_in, T = x_q.shape
    x_pad = np.pad(x_q.astype(np.int64), ((0, 0), (pad, pad)))
    idx   = np.arange(kernel)[None, :] + np.arange(T)[:, None]    # (T, K)
    x_unf = x_pad[:, idx]                                           # (C_in, T, K)
    out   = np.einsum('ctk,ock->ot', x_unf, w_q.astype(np.int64)) # (C_out, T)
    return sat16(out >> fb)                                          # (C_out, T)


def rmsnorm_hw_fast(x_q, w_norm_q, d_out, fb=FB):
    """Vectorised HW RMSNorm v2 — matches updated extract_itm_full.hw_rms_norm_inline.
    No pre-shift, raw x*x in int64, single final shift = log2_d + 2*fb - 1 - N."""
    log2_d      = int(round(math.log2(d_out)))
    x           = x_q.astype(np.int64)
    sq          = x * x                                              # (d_out, T)
    sum_d       = sq.sum(axis=0)                                     # (T,) int64
    total_shift = log2_d + 2*fb - 1 - RMS_EXTRA_PREC                 # N from extract_itm_full
    mean_i      = sum_d >> total_shift                               # (T,)
    S_t         = _RSQRT_ROM_NP[np.clip(mean_i, 0, 8191)]            # uses v2 ROM (imported)
    p1          = sat16((x * w_norm_q[:, None].astype(np.int64)) >> fb)
    return sat16((p1.astype(np.int64) * S_t[None, :]) >> fb)         # (d_out, T)


# ─────────────────────────────────────────────────────────────────────
#  Weight cache — quantise once, reuse for all test samples
# ─────────────────────────────────────────────────────────────────────

def cache_weights(model):
    """Returns (enc_cache, [block_cache, ...]) with pre-quantised int64 weights."""
    w_enc, b_enc = fuse_conv_bn(model.encoder[0], model.encoder[1])
    d_model = model.encoder[0].out_channels
    enc_cache = {
        'w_q'    : q(w_enc.reshape(d_model, 12)),
        'b_q'    : q(b_enc),
        'd_model': d_model,
    }

    block_caches = []
    for bk, blk in enumerate(model.layers):
        if not is_itm_block(blk):
            continue

        d_in  = blk.conv[0].in_channels
        d_out = blk.conv[0].out_channels
        dim   = d_out // 4
        inc   = blk.inception_block
        mixer = blk.mamba_block.mixer

        w_p1, b_p1       = fuse_conv_bn(blk.conv[0], blk.conv[1])
        scale_inc, shift_inc = fold_bn(inc.bn)

        d_inner = mixer.in_proj.weight.shape[0] // 2
        d_state = mixer.A_log.shape[1]
        dt_rank = mixer.dt_proj.in_features
        n_act   = dt_rank + 2 * d_state
        n_pad   = ((n_act + 15) // 16) * 16
        xpw     = to_f32(mixer.x_proj.weight)
        xp_pad  = np.zeros((n_pad, xpw.shape[1]), np.float32)
        xp_pad[:n_act] = xpw

        block_caches.append({
            'bk': bk, 'd_in': d_in, 'd_out': d_out, 'dim': dim,
            # P1
            'w_p1_q' : q(w_p1.reshape(d_out, d_in)),
            'b_p1_q' : q(b_p1),
            # Inception
            'w_bot_q': q(to_f32(inc.bottleneck.weight).reshape(dim, d_out)),
            'w_b1_q' : q(to_f32(inc.conv1.weight).reshape(dim, d_out)),
            'w_b2_q' : q(to_f32(inc.conv2.weight).reshape(dim, dim, -1)),
            'w_b3_q' : q(to_f32(inc.conv3.weight).reshape(dim, dim, -1)),
            'w_b4_q' : q(to_f32(inc.conv4.weight).reshape(dim, dim, -1)),
            'scale_q': q(scale_inc),
            'shift_q': q(shift_inc),
            # Mamba dims
            'd_inner': d_inner, 'd_state': d_state, 'dt_rank': dt_rank,
            # Mamba weights
            'w_norm_q': q(to_f32(blk.mamba_block.norm.weight)),
            'w_mx_q'  : q(to_f32(mixer.in_proj.weight[:d_inner, :])),
            'w_mz_q'  : q(to_f32(mixer.in_proj.weight[d_inner:, :])),
            'w_dw_q'  : q(to_f32(mixer.conv1d.weight)).reshape(d_inner, 4),
            'b_dw_q'  : q(to_f32(mixer.conv1d.bias)),
            'w_xp_q'  : q(xp_pad),
            'w_dt_q'  : q(to_f32(mixer.dt_proj.weight)),   # (d_inner, dt_rank)
            'b_dt_q'  : q(to_f32(mixer.dt_proj.bias)),     # (d_inner,)
            'A_q'     : q(to_f32(-torch.exp(mixer.A_log))), # (d_inner, d_state)
            'D_q'     : q(to_f32(mixer.D)),                 # (d_inner,)
            'w_out_q' : q(to_f32(mixer.out_proj.weight)),   # (d_in, d_inner)
        })

    return enc_cache, block_caches


def cache_weights_fb(model, fb):
    """Same as cache_weights but quantises at fractional-bit width fb."""
    def _q(arr): return q_fb(arr, fb)

    w_enc, b_enc = fuse_conv_bn(model.encoder[0], model.encoder[1])
    d_model = model.encoder[0].out_channels
    enc_cache = {
        'w_q': _q(w_enc.reshape(d_model, 12)),
        'b_q': _q(b_enc),
        'd_model': d_model,
    }

    block_caches = []
    for bk, blk in enumerate(model.layers):
        if not is_itm_block(blk):
            continue
        d_in  = blk.conv[0].in_channels
        d_out = blk.conv[0].out_channels
        dim   = d_out // 4
        inc   = blk.inception_block
        mixer = blk.mamba_block.mixer

        w_p1, b_p1       = fuse_conv_bn(blk.conv[0], blk.conv[1])
        scale_inc, shift_inc = fold_bn(inc.bn)

        d_inner = mixer.in_proj.weight.shape[0] // 2
        d_state = mixer.A_log.shape[1]
        dt_rank = mixer.dt_proj.in_features
        n_act   = dt_rank + 2 * d_state
        n_pad   = ((n_act + 15) // 16) * 16
        xpw     = to_f32(mixer.x_proj.weight)
        xp_pad  = np.zeros((n_pad, xpw.shape[1]), np.float32)
        xp_pad[:n_act] = xpw

        block_caches.append({
            'bk': bk, 'd_in': d_in, 'd_out': d_out, 'dim': dim,
            'w_p1_q' : _q(w_p1.reshape(d_out, d_in)),
            'b_p1_q' : _q(b_p1),
            'w_bot_q': _q(to_f32(inc.bottleneck.weight).reshape(dim, d_out)),
            'w_b1_q' : _q(to_f32(inc.conv1.weight).reshape(dim, d_out)),
            'w_b2_q' : _q(to_f32(inc.conv2.weight).reshape(dim, dim, -1)),
            'w_b3_q' : _q(to_f32(inc.conv3.weight).reshape(dim, dim, -1)),
            'w_b4_q' : _q(to_f32(inc.conv4.weight).reshape(dim, dim, -1)),
            'scale_q': _q(scale_inc),
            'shift_q': _q(shift_inc),
            'd_inner': d_inner, 'd_state': d_state, 'dt_rank': dt_rank,
            'w_norm_q': _q(to_f32(blk.mamba_block.norm.weight)),
            'w_mx_q'  : _q(to_f32(mixer.in_proj.weight[:d_inner, :])),
            'w_mz_q'  : _q(to_f32(mixer.in_proj.weight[d_inner:, :])),
            'w_dw_q'  : _q(to_f32(mixer.conv1d.weight)).reshape(d_inner, 4),
            'b_dw_q'  : _q(to_f32(mixer.conv1d.bias)),
            'w_xp_q'  : _q(xp_pad),
            'w_dt_q'  : _q(to_f32(mixer.dt_proj.weight)),
            'b_dt_q'  : _q(to_f32(mixer.dt_proj.bias)),
            'A_q'     : _q(to_f32(-torch.exp(mixer.A_log))),
            'D_q'     : _q(to_f32(mixer.D)),
            'w_out_q' : _q(to_f32(mixer.out_proj.weight)),
        })

    return enc_cache, block_caches


# ─────────────────────────────────────────────────────────────────────
#  Configurable-fb helpers  (used by fb11 / fb15 diagnostic variants)
# ─────────────────────────────────────────────────────────────────────

def _mmv_fb(x_q, w_q, fb):
    return sat16((w_q.astype(np.int64) @ x_q.astype(np.int64)) >> fb)


def _mulv_fb(x_q, y_q, fb):
    return sat16((x_q.astype(np.int64) * y_q.astype(np.int64)) >> fb)


def _hw_p1_fb(x_q, w_q, b_q, fb):
    return sat16(sat_add(_mmv_fb(x_q, w_q, fb), b_q[:, None]))


def _silu_fb(x_q, fb):
    if fb == FB:
        return lut_silu(x_q)
    s = 1 << fb
    xf = x_q.astype(np.float64) / s
    return sat16(np.floor(xf * (1.0 / (1.0 + np.exp(-xf))) * s).astype(np.int64))


def _softplus_fb(x_q, fb):
    if fb == FB:
        return lut_softplus(x_q)
    s = 1 << fb
    xf = x_q.astype(np.float64) / s
    return sat16(np.floor(np.log1p(np.exp(np.clip(xf, -20.0, 20.0))) * s).astype(np.int64))


def _exp_fb(x_q, fb):
    if fb == FB:
        return lut_exp(x_q)
    s = 1 << fb
    xf = x_q.astype(np.float64) / s
    return sat16(np.floor(np.exp(np.clip(xf, -20.0, 0.0)) * s).astype(np.int64))


_LUT_SILU_NP = np.asarray(_LUT_SILU, np.int64)
_LUT_SP_NP   = np.asarray(_LUT_SP,   np.int64)
_LUT_EXP_NP  = np.asarray(_LUT_EXP,  np.int64)


def _lut_lerp_apply(x_q, table_np, oor_lo_val, oor_hi_fn):
    """Generic linear-interp LUT lookup — RTL Option 1 exact.
       interp = lo + (sub * (hi - lo)) >> LUT_SHIFT
       where idx = (x - LUT_LO) >> LUT_SHIFT  and  sub = lower LUT_SHIFT bits."""
    LUT_LO    = -8 * SCALE          # -16384 for FB=11
    LUT_SHIFT = FB - 4              # 7 for FB=11
    x         = np.asarray(x_q, np.int64)
    in_range  = (x >= LUT_LO) & (x < -LUT_LO)
    x_off     = x - LUT_LO
    idx       = np.clip(np.where(in_range, x_off >> LUT_SHIFT, np.int64(0)),
                        0, 254)     # guard: idx+1 <= 255
    sub       = np.where(in_range, x_off & ((1 << LUT_SHIFT) - 1), np.int64(0))
    lo        = table_np[idx]
    hi        = table_np[idx + 1]
    interp    = sat16(lo + ((sub * (hi - lo)) >> LUT_SHIFT))
    oor       = np.where(x < LUT_LO, oor_lo_val, oor_hi_fn(x))
    return np.where(in_range, interp, oor)


def lut_silu_lerp(x_q):
    return _lut_lerp_apply(x_q, _LUT_SILU_NP, np.int64(0), sat16)

def lut_softplus_lerp(x_q):
    return _lut_lerp_apply(x_q, _LUT_SP_NP, np.int64(0), sat16)

def lut_exp_lerp(x_q):
    return _lut_lerp_apply(x_q, _LUT_EXP_NP, np.int64(0), lambda x: np.int64(32767))


def _silu_fb_lerp(x_q, fb):
    if fb == FB:
        return lut_silu_lerp(x_q)
    return _silu_fb(x_q, fb)

def _softplus_fb_lerp(x_q, fb):
    if fb == FB:
        return lut_softplus_lerp(x_q)
    return _softplus_fb(x_q, fb)

def _exp_fb_lerp(x_q, fb):
    if fb == FB:
        return lut_exp_lerp(x_q)
    return _exp_fb(x_q, fb)


# ── Float nonlinear helpers (use SCALE, Q4.11-exact I/O, float arithmetic) ──
def _silu_float(x_q):
    xf = x_q.astype(np.float64) / SCALE
    return sat16(np.floor(xf / (1.0 + np.exp(-xf)) * SCALE).astype(np.int64))

def _softplus_float(x_q):
    xf = x_q.astype(np.float64) / SCALE
    return sat16(np.floor(np.log1p(np.exp(np.clip(xf, -20.0, 20.0))) * SCALE).astype(np.int64))

def _exp_float(x_q):
    xf = x_q.astype(np.float64) / SCALE
    return sat16(np.floor(np.exp(np.clip(xf, -20.0, 0.0)) * SCALE).astype(np.int64))

def _rmsnorm_float(x_q, w_norm_q, d_out):
    xf = x_q.astype(np.float64) / SCALE
    wf = w_norm_q.astype(np.float64) / SCALE
    rms = np.sqrt((xf * xf).mean(axis=0, keepdims=True) + 1e-6)
    return sat16(np.floor(xf * wf[:, None] / rms * SCALE).astype(np.int64))


def _rmsnorm_fb(x_q, w_norm_q, d_out, fb):
    if fb == FB:
        return rmsnorm_hw_fast(x_q, w_norm_q, d_out)
    s = 1 << fb
    xf = x_q.astype(np.float64) / s
    wf = w_norm_q.astype(np.float64) / s
    rms = np.sqrt((xf * xf).mean(axis=0, keepdims=True) + 1e-6)
    return sat16(np.floor(xf * wf[:, None] / rms * s).astype(np.int64))


def _rmsnorm_fb_v2(x_q, w_norm_q, d_out, fb):
    """RMSNorm v2 wrapper — uses rmsnorm_hw_v2 when fb==FB, else float fallback."""
    if fb == FB:
        return rmsnorm_hw_v2(x_q, w_norm_q, d_out, fb=fb, extra_prec=6)
    s = 1 << fb
    xf = x_q.astype(np.float64) / s
    wf = w_norm_q.astype(np.float64) / s
    rms = np.sqrt((xf * xf).mean(axis=0, keepdims=True) + 1e-6)
    return sat16(np.floor(xf * wf[:, None] / rms * s).astype(np.int64))


def infer_mamba_hw_cfg(c, p1_q, fb, use_lerp_exp=False, lerp_lut_all=False,
                       float_nl=False, float_ssm=False,
                       float_rms=False, float_silu=False,
                       float_softplus=False, float_exp=False,
                       rms_v2=False):
    """Mamba inference with configurable fractional bits. RTL-exact when fb==FB.

    Per-nonlinear bisection flags:
      float_rms       : RMSNorm uses float64 (Q4.11 I/O)
      float_silu      : SiLU uses float64
      float_softplus  : softplus uses float64
      float_exp       : exp (SSM scan) uses float64
      float_nl        : shortcut → all 4 above = True

    LUT improvement flags (only applied when corresponding float_* is False):
      use_lerp_exp    : exp uses lerped LUT (256-entry table, 128-step sub-interp)
      lerp_lut_all    : silu + softplus + exp all use lerped LUT

    Integer RMSNorm improvement:
      rms_v2          : use rmsnorm_hw_v2 (no pre-shift + finer ROM, N=6 extra prec).
                        Ignored if float_rms=True. Expected to recover most RMSNorm AUC drop.

    float_ssm=True   : SSM scan inner loop uses float64 (no integer truncation per step)
    """
    # ── expand float_nl shortcut ────────────────────────────────────
    if float_nl:
        float_rms = float_silu = float_softplus = float_exp = True

    # ── select RMSNorm ──────────────────────────────────────────────
    if float_rms:
        rms_fn = _rmsnorm_float
    elif rms_v2:
        rms_fn = lambda xq, wq, d: _rmsnorm_fb_v2(xq, wq, d, fb)
    else:
        rms_fn = lambda xq, wq, d: _rmsnorm_fb(xq, wq, d, fb)

    # ── select SiLU ─────────────────────────────────────────────────
    if float_silu:
        silu_fn = _silu_float
    elif lerp_lut_all:
        silu_fn = lambda xq: _silu_fb_lerp(xq, fb)
    else:
        silu_fn = lambda xq: _silu_fb(xq, fb)

    # ── select Softplus ─────────────────────────────────────────────
    if float_softplus:
        softplus_fn = _softplus_float
    elif lerp_lut_all:
        softplus_fn = lambda xq: _softplus_fb_lerp(xq, fb)
    else:
        softplus_fn = lambda xq: _softplus_fb(xq, fb)

    # ── select Exp (used in SSM scan) ───────────────────────────────
    if float_exp:
        exp_ssm_fn = _exp_float
    elif lerp_lut_all or use_lerp_exp:
        exp_ssm_fn = lambda xq: _exp_fb_lerp(xq, fb)
    else:
        exp_ssm_fn = lambda xq: _exp_fb(xq, fb)

    d_in    = p1_q.shape[0]
    T       = p1_q.shape[1]
    d_inner = c['d_inner']
    d_state = c['d_state']
    dt_rank = c['dt_rank']

    p1_norm_q = rms_fn(p1_q, c['w_norm_q'], d_in)
    x_inner_q = _mmv_fb(p1_norm_q, c['w_mx_q'], fb)
    z_gate_q  = _mmv_fb(p1_norm_q, c['w_mz_q'], fb)

    x_pad_dw = np.pad(x_inner_q.astype(np.int64), ((0, 0), (3, 0)))
    idx_dw   = np.arange(4)[None, :] + np.arange(T)[:, None]
    x_unf_dw = x_pad_dw[:, idx_dw]
    x_conv_q = np.einsum('dtk,dk->dt', x_unf_dw, c['w_dw_q'].astype(np.int64))
    x_conv_q = sat_add(sat16(x_conv_q >> fb), c['b_dw_q'][:, None])

    u_q = silu_fn(x_conv_q)
    xproj_q = _mmv_fb(u_q, c['w_xp_q'], fb)

    dt_raw_q = xproj_q[:dt_rank, :]
    delta_q  = sat16((c['w_dt_q'].astype(np.int64) @ dt_raw_q.astype(np.int64)) >> fb)
    delta_q  = softplus_fn(sat_add(delta_q, c['b_dt_q'][:, None]))

    B_q = xproj_q[dt_rank:dt_rank + d_state, :]
    C_q = xproj_q[dt_rank + d_state:dt_rank + 2*d_state, :]
    A_q = c['A_q']
    D_q = c['D_q']

    if float_ssm:
        # SSM scan in float64 — uses quantized Q4.11 inputs, removes integer >> truncation
        s    = float(1 << fb)
        h_f  = np.zeros((d_inner, d_state), np.float64)
        y_ssm_q = np.zeros((d_inner, T), np.int64)
        u_f  = u_q.astype(np.float64) / s
        A_f  = A_q.astype(np.float64) / s
        D_f  = D_q.astype(np.float64) / s
        for t in range(T):
            dt_f  = delta_q[:, t:t+1].astype(np.float64) / s
            dA_f  = np.exp(np.clip(dt_f * A_f, -30.0, 0.0))
            dB_f  = dt_f * (B_q[:, t].astype(np.float64) / s)[np.newaxis, :]
            h_f   = dA_f * h_f + dB_f * u_f[:, t:t+1]
            y_ch_f = h_f @ (C_q[:, t].astype(np.float64) / s)
            du_f   = D_f * u_f[:, t]
            y_ssm_q[:, t] = sat16(np.floor((y_ch_f + du_f) * s).astype(np.int64))
    else:
        h_q     = np.zeros((d_inner, d_state), np.int64)
        y_ssm_q = np.zeros((d_inner, T), np.int64)
        for t in range(T):
            dt_t  = delta_q[:, t:t+1]
            da_in = sat16((dt_t * A_q.astype(np.int64)) >> fb)
            dA    = exp_ssm_fn(da_in)
            dB    = sat16((dt_t * B_q[:, t].astype(np.int64)[np.newaxis, :]) >> fb)
            t1    = sat16((dA.astype(np.int64) * h_q) >> fb)
            t2    = sat16((dB.astype(np.int64) * u_q[:, t:t+1]) >> fb)
            h_q   = sat_add(t1, t2)
            y_ch  = sat16((h_q.astype(np.int64) @ C_q[:, t].astype(np.int64)) >> fb)
            du    = sat16((D_q.astype(np.int64) * u_q[:, t].astype(np.int64)) >> fb)
            y_ssm_q[:, t] = sat_add(y_ch, du)

    y_gated_q = _mulv_fb(y_ssm_q, silu_fn(z_gate_q), fb)
    return _mmv_fb(y_gated_q, c['w_out_q'], fb)


def infer_block_hw_cfg(c, x_q, fb, py_formula=True, **mamba_kwargs):
    """ITMBlock integer inference with configurable fractional bits.
    mamba_kwargs is passed through to infer_mamba_hw_cfg.

    py_formula=True  (default, PyTorch ITMBlock.forward — matches RTL after fix):
        x1 = relu(bn(inc_cat))   — matches BaseInceptionBlock output
        x2 = relu(mam)           — matches ITMBlock self.relu(self.mamba_block(...))
        out = sat_add(x1, x2)
    py_formula=False (legacy HW workaround): out = relu(bn(inc + mam))
    """
    T = x_q.shape[1]

    p1_q    = _hw_p1_fb(x_q, c['w_p1_q'], c['b_p1_q'], fb)
    bot_q   = _mmv_fb(p1_q, c['w_bot_q'], fb)

    x_pad3  = np.pad(p1_q.astype(np.int64), ((0, 0), (1, 1)), mode='edge')
    p1_mp_q = np.maximum(np.maximum(x_pad3[:, :T], x_pad3[:, 1:T+1]), x_pad3[:, 2:T+2])
    b1_q    = _mmv_fb(p1_mp_q, c['w_b1_q'], fb)

    b2_q = pe_mac_conv_fast(bot_q, c['w_b2_q'], kernel=9,  pad=4,  fb=fb)
    b3_q = pe_mac_conv_fast(bot_q, c['w_b3_q'], kernel=19, pad=9,  fb=fb)
    b4_q = pe_mac_conv_fast(bot_q, c['w_b4_q'], kernel=39, pad=19, fb=fb)

    inc_cat_q   = np.concatenate([b1_q, b2_q, b3_q, b4_q], axis=0)
    mamba_out_q = infer_mamba_hw_cfg(c, p1_q, fb, **mamba_kwargs)

    if py_formula:
        # PyTorch formula: relu(bn(inc)) + relu(mam)
        mul_i = inc_cat_q.astype(np.int64) * c['scale_q'][:, None].astype(np.int64)
        bn_i  = sat16(sat16(mul_i >> fb) + c['shift_q'][:, None].astype(np.int64))
        x1    = np.where(bn_i < 0, np.int64(0), bn_i)
        x2    = np.where(mamba_out_q < 0, np.int64(0), mamba_out_q)
        return sat_add(x1, x2)

    # HW workaround formula: relu(bn(inc + mam))
    raw = sat_add(inc_cat_q, mamba_out_q)
    mul = raw.astype(np.int64) * c['scale_q'][:, None].astype(np.int64)
    bn  = sat16(sat16(mul >> fb) + c['shift_q'][:, None].astype(np.int64))
    return np.where(bn < 0, np.int64(0), bn)


def infer_block_float_mamba(c, blk, x_q, dev):
    """Integer inception + float Mamba. Combined via RTL formula bn_relu(inc+mamba)."""
    T = x_q.shape[1]

    p1_q    = hw_p1(x_q, c['w_p1_q'], c['b_p1_q'])
    bot_q   = pe_mac_mv(p1_q, c['w_bot_q'])

    x_pad3  = np.pad(p1_q.astype(np.int64), ((0, 0), (1, 1)), mode='edge')
    p1_mp_q = np.maximum(np.maximum(x_pad3[:, :T], x_pad3[:, 1:T+1]), x_pad3[:, 2:T+2])
    b1_q    = pe_mac_mv(p1_mp_q, c['w_b1_q'])

    b2_q = pe_mac_conv_fast(bot_q, c['w_b2_q'], kernel=9,  pad=4)
    b3_q = pe_mac_conv_fast(bot_q, c['w_b3_q'], kernel=19, pad=9)
    b4_q = pe_mac_conv_fast(bot_q, c['w_b4_q'], kernel=39, pad=19)
    inc_cat_q = np.concatenate([b1_q, b2_q, b3_q, b4_q], axis=0)

    # Float Mamba: dequant p1_q → (1, T, d_out) → mamba_block → relu → Q9.7
    p1_f = torch.tensor(
        (p1_q.astype(np.float32) / SCALE).T[np.newaxis], dtype=torch.float32
    ).to(dev)                                                      # (1, T, d_out)
    with torch.no_grad():
        mam_f = torch.relu(blk.mamba_block(p1_f))                # (1, T, d_out)
    mamba_out_q = q(mam_f[0].cpu().numpy().T)                    # (d_out, T)

    raw = sat_add(inc_cat_q, mamba_out_q)
    mul = raw.astype(np.int64) * c['scale_q'][:, None].astype(np.int64)
    bn  = sat16(sat16(mul >> FB) + c['shift_q'][:, None].astype(np.int64))
    return np.where(bn < 0, np.int64(0), bn)


# ─────────────────────────────────────────────────────────────────────
#  Integer Mamba inference
# ─────────────────────────────────────────────────────────────────────

def infer_mamba_hw(c, p1_q):
    d_in    = p1_q.shape[0]
    T       = p1_q.shape[1]
    d_inner = c['d_inner']
    d_state = c['d_state']
    dt_rank = c['dt_rank']

    # RMSNorm — vectorised over T
    p1_norm_q = rmsnorm_hw_fast(p1_q, c['w_norm_q'], d_in)

    # M1a/M1b: in_proj — pe_mac_mv handles (C_in, T) as matrix multiply
    x_inner_q = pe_mac_mv(p1_norm_q, c['w_mx_q'])   # (d_inner, T)
    z_gate_q  = pe_mac_mv(p1_norm_q, c['w_mz_q'])   # (d_inner, T)

    # M2: depthwise causal conv1d k=4 — vectorised via einsum
    # pad 3 zeros on left (causal), no right pad
    x_pad_dw = np.pad(x_inner_q.astype(np.int64), ((0, 0), (3, 0)))
    idx_dw   = np.arange(4)[None, :] + np.arange(T)[:, None]        # (T, 4)
    x_unf_dw = x_pad_dw[:, idx_dw]                                   # (d_inner, T, 4)
    x_conv_q = np.einsum('dtk,dk->dt', x_unf_dw, c['w_dw_q'].astype(np.int64))
    x_conv_q = sat_add(sat16(x_conv_q >> FB), c['b_dw_q'][:, None])

    # M3: SiLU
    u_q = lut_silu(x_conv_q)                         # (d_inner, T)

    # M4: x_proj — matrix multiply over T
    xproj_q = pe_mac_mv(u_q, c['w_xp_q'])            # (n_pad, T)

    # M5: dt_proj + bias + softplus — fully vectorised over T
    dt_raw_q = xproj_q[:dt_rank, :]                   # (dt_rank, T)
    delta_q  = sat16(
        (c['w_dt_q'].astype(np.int64) @ dt_raw_q.astype(np.int64)) >> FB
    )                                                  # (d_inner, T)
    delta_q  = lut_softplus(sat_add(delta_q, c['b_dt_q'][:, None]))

    # M6: SSM scan — sequential over T (causal), vectorised over d_state
    B_q     = xproj_q[dt_rank:dt_rank + d_state, :]          # (d_state, T)
    C_q     = xproj_q[dt_rank + d_state:dt_rank + 2*d_state, :]  # (d_state, T)
    A_q     = c['A_q']                                        # (d_inner, d_state)
    D_q     = c['D_q']                                        # (d_inner,)
    h_q     = np.zeros((d_inner, d_state), np.int64)
    y_ssm_q = np.zeros((d_inner, T), np.int64)

    for t in range(T):
        dt_t = delta_q[:, t:t+1]                              # (d_inner, 1)

        # dA: exp(delta × A) — (d_inner, d_state)
        da_in = sat16((dt_t * A_q.astype(np.int64)) >> FB)
        dA    = lut_exp(da_in)

        # dB: delta × B[t] — (d_inner, d_state)
        dB = sat16((dt_t * B_q[:, t].astype(np.int64)[np.newaxis, :]) >> FB)

        # h = dA*h + dB*u  — (d_inner, d_state)
        t1  = sat16((dA.astype(np.int64) * h_q) >> FB)
        t2  = sat16((dB.astype(np.int64) * u_q[:, t:t+1]) >> FB)
        h_q = sat_add(t1, t2)

        # y = h @ C[t] + D*u  (int64 accumulation before >>fb, matches RTL grouping)
        y_ch = sat16((h_q.astype(np.int64) @ C_q[:, t].astype(np.int64)) >> FB)
        du   = sat16((D_q.astype(np.int64) * u_q[:, t].astype(np.int64)) >> FB)
        y_ssm_q[:, t] = sat_add(y_ch, du)

    # M7: y_gated = y_ssm × SiLU(z)
    y_gated_q = pe_mul_vec(y_ssm_q, lut_silu(z_gate_q))

    # M8: out_proj
    return pe_mac_mv(y_gated_q, c['w_out_q'])                # (d_in, T)


# ─────────────────────────────────────────────────────────────────────
#  Integer inference — one ITMBlock
# ─────────────────────────────────────────────────────────────────────

def infer_block_hw(c, x_q):
    """Integer inference for one ITMBlock. Returns final_q (d_out, T)."""
    d_out = c['d_out']
    T     = x_q.shape[1]

    # P1: Conv+BN fused
    p1_q = hw_p1(x_q, c['w_p1_q'], c['b_p1_q'])              # (d_out, T)

    # Bottleneck
    bot_q = pe_mac_mv(p1_q, c['w_bot_q'])                     # (dim, T)

    # B1: MaxPool1d(k=3, s=1, pad=1) then conv1 — vectorised max
    x_pad3  = np.pad(p1_q.astype(np.int64), ((0, 0), (1, 1)), mode='edge')
    p1_mp_q = np.maximum(
        np.maximum(x_pad3[:, :T], x_pad3[:, 1:T+1]),
        x_pad3[:, 2:T+2]
    )
    b1_q = pe_mac_mv(p1_mp_q, c['w_b1_q'])                    # (dim, T)

    # B2/B3/B4: dilated convs on bot_q — vectorised einsum
    b2_q = pe_mac_conv_fast(bot_q, c['w_b2_q'], kernel=9,  pad=4)
    b3_q = pe_mac_conv_fast(bot_q, c['w_b3_q'], kernel=19, pad=9)
    b4_q = pe_mac_conv_fast(bot_q, c['w_b4_q'], kernel=39, pad=19)

    inc_cat_q = np.concatenate([b1_q, b2_q, b3_q, b4_q], axis=0)  # (d_out, T)

    # Mamba branch
    mamba_out_q = infer_mamba_hw(c, p1_q)                     # (d_out, T)

    # Final: relu(bn(inc_cat)) + relu(mamba_out)  (PyTorch ITMBlock formula)
    mul_i = inc_cat_q.astype(np.int64) * c['scale_q'][:, None].astype(np.int64)
    bn_i  = sat16(sat16(mul_i >> FB) + c['shift_q'][:, None].astype(np.int64))
    x1    = np.where(bn_i < 0, np.int64(0), bn_i)
    x2    = np.where(mamba_out_q < 0, np.int64(0), mamba_out_q)
    return sat_add(x1, x2)                                    # (d_out, T)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_type', default='super')
    ap.add_argument('--variant', default='hw',
                    choices=['hw', 'ideal_q97', 'float_enc', 'float_mamba',
                             'fb11', 'fb15', 'fb11_lerp', 'fb11_lerp_all',
                             'fb11_fn', 'fb11_fssm', 'fb11_fn_fssm',
                             'fb11_frms', 'fb11_fsilu', 'fb11_fsp', 'fb11_fexp',
                             'fb11_rmsv2',
                             'fb11_py', 'fb11_fn_py', 'fb11_rmsv2_py',
                             'all'],
                    help='Variant(s) to evaluate (all = run everything)')
    args = ap.parse_args()

    ALL_VARIANTS = ['hw', 'ideal_q97', 'float_enc', 'float_mamba',
                    'fb11', 'fb15', 'fb11_lerp', 'fb11_lerp_all',
                    'fb11_fn', 'fb11_fssm', 'fb11_fn_fssm',
                    'fb11_frms', 'fb11_fsilu', 'fb11_fsp', 'fb11_fexp',
                    'fb11_rmsv2',
                    'fb11_py', 'fb11_fn_py', 'fb11_rmsv2_py']
    run = set(ALL_VARIANTS if args.variant == 'all' else [args.variant])

    params = get_config('config.yaml')
    params['exp_type'] = args.exp_type.lower()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[MODEL] Loading...')
    _, _, test_loader_batch, num_class, _ = get_loaders(
        params['data'], params['exp_type'],
        params['hyperparameters']['batch_size'])
    _, _, test_loader_single, _, _ = get_loaders(
        params['data'], params['exp_type'], batch_size=1)

    model = ITMN(n_classes=num_class, **params['model']).to(dev)
    model.load_state_dict(
        torch.load(params['test_ckpt_path'], map_location=dev)['model_state_dict'])
    model.eval()

    print('\n[WEIGHTS] Caching Q9.7 weights...')
    enc_cache, block_caches = cache_weights(model)
    itm_indices = [c['bk'] for c in block_caches]

    has_maxpool = [False]
    for idx in range(1, len(itm_indices)):
        prev_bk, cur_bk = itm_indices[idx-1], itm_indices[idx]
        mp = any(isinstance(model.layers[j], torch.nn.MaxPool1d)
                 for j in range(prev_bk + 1, cur_bk))
        has_maxpool.append(mp)
    print(f"  ITMBlock indices : {itm_indices}")
    print(f"  MaxPool before   : {has_maxpool}")

    # ── Float baseline + ideal_q97 in a single pass ───────────────────
    print('\n[FLOAT] Running float baseline...')
    _ibuf: dict = {}
    if 'ideal_q97' in run:
        hook_h = model.layers[itm_indices[-1]].register_forward_hook(
            lambda m, i, o: _ibuf.update({'x': o.detach().cpu().numpy()})
        )

    float_preds, ideal_preds, targets = [], [], []
    for sample in tqdm.tqdm(test_loader_batch, desc='Float'):
        wav_t, lbl = sample['waveform'].to(dev), sample['label']
        with torch.no_grad():
            prob = torch.sigmoid(model(wav_t)).cpu().numpy()   # (B, C)
        float_preds.extend(prob.tolist())
        targets.extend(lbl.numpy().tolist())

        if 'ideal_q97' in run:
            x_out = _ibuf['x']                                 # (B, d_out, T)
            for b in range(x_out.shape[0]):
                xq = q(x_out[b])                               # (d_out, T) Q9.7
                hm = torch.tensor(
                    xq.astype(np.float32).mean(-1) / SCALE,
                    dtype=torch.float32).unsqueeze(0).to(dev)
                with torch.no_grad():
                    p = torch.sigmoid(model.classifier(hm)).cpu().numpy()[0]
                ideal_preds.append(p.tolist())

    if 'ideal_q97' in run:
        hook_h.remove()

    target_arr = np.array(targets)
    float_m    = calculate_metrics(target_arr, np.array(float_preds))
    print(f"  Float     : AUC={float_m['AUC']:.4f} | TPR={float_m['TPR']:.4f}")
    results = {'float': float_m}

    if 'ideal_q97' in run:
        ideal_m = calculate_metrics(target_arr, np.array(ideal_preds))
        results['ideal_q97'] = ideal_m
        print(f"  ideal_q97 : AUC={ideal_m['AUC']:.4f} | TPR={ideal_m['TPR']:.4f}  "
              f"(gap={ideal_m['AUC'] - float_m['AUC']:+.4f})")

    # ── Generic runner: iterates test_loader_single with given enc/blk fns ──
    def _run_chain(desc, enc_fn, blk_fn):
        preds = []
        for sample in tqdm.tqdm(test_loader_single, desc=desc):
            wav = sample['waveform'][0].numpy()   # (T, 12)
            x_q = enc_fn(wav)
            for idx, c in enumerate(block_caches):
                if has_maxpool[idx]:
                    x_q = hw_maxpool(x_q)
                x_q = blk_fn(c, x_q)
            hm = torch.tensor(
                x_q.astype(np.float32).mean(-1) / SCALE,
                dtype=torch.float32).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = torch.sigmoid(model.classifier(hm)).cpu().numpy()[0]
            preds.append(p.tolist())
        return calculate_metrics(target_arr, np.array(preds))

    # ── hw: RTL-exact Q9.7 ───────────────────────────────────────────
    if 'hw' in run:
        def _enc_hw(wav):
            return hw_p1(q(wav.T), enc_cache['w_q'], enc_cache['b_q'])
        hw_m = _run_chain('[HW]   Q9.7', _enc_hw, infer_block_hw)
        results['hw'] = hw_m
        print(f"  hw        : AUC={hw_m['AUC']:.4f} | TPR={hw_m['TPR']:.4f}  "
              f"(gap={hw_m['AUC'] - float_m['AUC']:+.4f})")

    # ── float_enc: float encoder → Q9.7 → integer blocks ─────────────
    if 'float_enc' in run:
        def _enc_float(wav):
            wt = torch.tensor(wav.T[np.newaxis], dtype=torch.float32).to(dev)
            with torch.no_grad():
                xe = model.encoder(wt)            # (1, d_model, T)
            return q(xe[0].cpu().numpy())          # Q9.7  (d_model, T)
        fe_m = _run_chain('[FE]   float_enc', _enc_float, infer_block_hw)
        results['float_enc'] = fe_m
        print(f"  float_enc : AUC={fe_m['AUC']:.4f} | TPR={fe_m['TPR']:.4f}  "
              f"(gap={fe_m['AUC'] - float_m['AUC']:+.4f})")

    # ── float_mamba: integer inception + float Mamba ──────────────────
    if 'float_mamba' in run:
        def _enc_hw2(wav):
            return hw_p1(q(wav.T), enc_cache['w_q'], enc_cache['b_q'])
        def _blk_fm(c, x_q):
            return infer_block_float_mamba(c, model.layers[c['bk']], x_q, dev)
        fm_m = _run_chain('[FM]   float_mamba', _enc_hw2, _blk_fm)
        results['float_mamba'] = fm_m
        print(f"  float_mamba: AUC={fm_m['AUC']:.4f} | TPR={fm_m['TPR']:.4f}  "
              f"(gap={fm_m['AUC'] - float_m['AUC']:+.4f})")

    # ── fb variants: integer chain with configurable float overrides ─────
    # Each entry: (fb_val, label, kwargs for infer_block_hw_cfg)
    # float_nl=True  → silu/softplus/exp/rmsnorm use float64 (no LUT/ROM error)
    # float_ssm=True → SSM scan inner loop uses float64 (no >> truncation per step)
    FB_VARIANTS = [
        # baseline & wider-FB ceiling
        (11, 'fb11',         dict()),
        (15, 'fb15',         dict()),
        # LUT lerp variants (RTL Option 1)
        (11, 'fb11_lerp',    dict(use_lerp_exp=True)),                  # exp only (legacy)
        (11, 'fb11_lerp_all',dict(lerp_lut_all=True)),                  # silu+softplus+exp
        # high-precision ceilings
        (11, 'fb11_fn',      dict(float_nl=True)),                       # all 4 NL = float
        (11, 'fb11_fssm',    dict(float_ssm=True)),                      # SSM scan = float
        (11, 'fb11_fn_fssm', dict(float_nl=True, float_ssm=True)),       # both
        # per-nonlinear bisection (isolate worst LUT/ROM)
        (11, 'fb11_frms',    dict(float_rms=True)),                      # only RMSNorm float
        (11, 'fb11_fsilu',   dict(float_silu=True)),                     # only SiLU float
        (11, 'fb11_fsp',     dict(float_softplus=True)),                 # only Softplus float
        (11, 'fb11_fexp',    dict(float_exp=True)),                      # only exp float
        # Integer RMSNorm v2: no-pre-shift + finer ROM (N=6). Should match fb11_frms (~0.86)
        (11, 'fb11_rmsv2',   dict(rms_v2=True)),                         # integer RMSNorm v2
        # PyTorch-formula final stage variants — kept for regression vs legacy HW workaround
        # (py_formula is now the default; these are explicit-flag duplicates of fb11/fb11_fn/fb11_rmsv2).
        (11, 'fb11_py',      dict(py_formula=True)),                     # PyTorch formula only
        (11, 'fb11_fn_py',   dict(py_formula=True, float_nl=True)),      # + float NL (ceiling)
        (11, 'fb11_rmsv2_py',dict(py_formula=True, rms_v2=True)),        # + integer RMSNorm v2
    ]
    _fb_caches = {}  # fb_val -> (enc_fb, blk_fb)
    for fb_val, label, cfg in FB_VARIANTS:
        if label not in run:
            continue
        if fb_val not in _fb_caches:
            print(f'\n[WEIGHTS] Caching fb{fb_val} weights...')
            _fb_caches[fb_val] = cache_weights_fb(model, fb_val)
        enc_fb, blk_fb = _fb_caches[fb_val]

        def _enc_fb_fn(wav, _ec=enc_fb, _fv=fb_val):
            return _hw_p1_fb(q_fb(wav.T, _fv), _ec['w_q'], _ec['b_q'], _fv)

        def _blk_fb_fn(c, x_q, _bc=blk_fb, _fv=fb_val, _cfg=cfg):
            cc = next(b for b in _bc if b['bk'] == c['bk'])
            return infer_block_hw_cfg(cc, x_q, _fv, **_cfg)

        fb_m = _run_chain(f'[{label.upper()}]  {label}', _enc_fb_fn, _blk_fb_fn)
        results[label] = fb_m
        print(f"  {label:<14}: AUC={fb_m['AUC']:.4f} | TPR={fb_m['TPR']:.4f}  "
              f"(gap={fb_m['AUC'] - float_m['AUC']:+.4f})")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"{'Variant':<14} {'AUC':>7} {'TPR':>7} {'AUC gap':>10}")
    print(f"{'─'*62}")
    for name, m in results.items():
        gap = f"{m['AUC'] - float_m['AUC']:+.4f}" if name != 'float' else '      ─'
        print(f"{name:<14} {m['AUC']:>7.4f} {m['TPR']:>7.4f} {gap:>10}")
    print(f"{'='*62}")


if __name__ == '__main__':
    main()
