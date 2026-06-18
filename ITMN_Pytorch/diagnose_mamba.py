"""
diagnose_mamba.py — Step-by-step comparison of infer_mamba_hw (test_hw)
vs extract_mamba_hwexact (extract_itm_full) on block 0 of the first test sample.

Prints per-stage mismatch counts to identify where divergence begins.
"""
import math, sys
import numpy as np
import torch
from pathlib import Path

from extract_itm_full import (
    q, to_f32, sat16, sat_add, FB, SCALE,
    fuse_conv_bn, fold_bn, hw_p1, is_itm_block,
    hw_rms_norm_inline, pe_mac_mv, pe_mul_vec,
    lut_silu, lut_softplus, lut_exp,
    _RSQRT_ROM, RMS_EXTRA_PREC,
)
from test_hw import rmsnorm_hw_fast, pe_mac_conv_fast, cache_weights
from dataset import get_loaders
from ecg_models.ITMN import ITMN
from utils.utils import get_config

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def chk(label, a, b):
    if np.array_equal(a.ravel(), b.ravel()):
        print(f"  {label:<35s}  {PASS}")
        return True
    diff = np.abs(a.ravel().astype(np.int64) - b.ravel().astype(np.int64))
    print(f"  {label:<35s}  {FAIL}  n={int((diff>0).sum())}/{diff.size}  max={diff.max()}")
    return False


def mamba_stages_extract(mixer, p1_q, fb=FB):
    """Run extract_mamba_hwexact step by step, return dict of intermediates."""
    d_inner = mixer.in_proj.weight.shape[0] // 2
    d_state = mixer.A_log.shape[1]
    dt_rank = mixer.dt_proj.in_features
    d_in    = mixer.in_proj.weight.shape[1]
    T       = p1_q.shape[1]

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

    # find norm weight from parent mamba_block
    # (passed separately — caller provides it via w_norm_q)
    return dict(
        w_mx_q=w_mx_q, w_mz_q=w_mz_q, w_dw_q=w_dw_q, b_dw_q=b_dw_q,
        w_xp_q=w_xp_q, w_dt_q=w_dt_q, b_dt_q=b_dt_q,
        A_q=A_q, D_q=D_q, w_out_q=w_out_q,
        d_inner=d_inner, d_state=d_state, dt_rank=dt_rank, T=T,
    )


def run_mamba_extract(p1_q, w_norm_q, W, fb=FB):
    """hw_rms_norm_inline path."""
    d_in    = p1_q.shape[0]
    T       = W['T']
    d_inner = W['d_inner']
    d_state = W['d_state']
    dt_rank = W['dt_rank']

    p1_norm = hw_rms_norm_inline(p1_q, w_norm_q, d_in, fb=fb)
    x_inner = pe_mac_mv(p1_norm, W['w_mx_q'])
    z_gate  = pe_mac_mv(p1_norm, W['w_mz_q'])

    # M2 conv
    x_conv_q = np.zeros((d_inner, T), np.int64)
    for d in range(d_inner):
        for t in range(T):
            acc = np.int64(0)
            for k in range(4):
                t_eff = t + k - 3
                if 0 <= t_eff < T:
                    acc += x_inner[d, t_eff] * W['w_dw_q'][d, k]
            x_conv_q[d, t] = acc
    x_conv = sat_add(sat16(x_conv_q >> fb), W['b_dw_q'][:, None])

    u = lut_silu(x_conv)
    xproj = pe_mac_mv(u, W['w_xp_q'])

    dt_raw = xproj[:dt_rank, :]
    delta = np.zeros((d_inner, T), np.int64)
    for t in range(T):
        for g in range(d_inner // 16):
            acc = np.zeros(16, np.int64)
            for r in range(dt_rank):
                acc += dt_raw[r, t] * W['w_dt_q'][g*16:(g+1)*16, r].astype(np.int64)
            delta[g*16:(g+1)*16, t] = sat_add(sat16(acc >> fb), W['b_dt_q'][g*16:(g+1)*16])
    delta = lut_softplus(delta)

    B = xproj[dt_rank:dt_rank+d_state, :]
    C = xproj[dt_rank+d_state:dt_rank+2*d_state, :]
    h = np.zeros((d_inner, d_state), np.int64)
    y_ssm = np.zeros((d_inner, T), np.int64)

    for t in range(T):
        for s in range(d_state):
            da_in = pe_mul_vec(delta[:, t], W['A_q'][:, s])
            dA    = lut_exp(da_in)
            dB    = pe_mul_vec(delta[:, t], np.full(d_inner, B[s, t], np.int64))
            t1    = pe_mul_vec(dA, h[:, s])
            t2    = pe_mul_vec(dB, u[:, t])
            h[:, s] = sat_add(t1, t2)
        for g in range(d_inner // 16):
            acc = np.zeros(16, np.int64)
            for s in range(d_state):
                acc += C[s, t] * h[g*16:(g+1)*16, s].astype(np.int64)
            y_ch = sat16(acc >> fb)
            du   = pe_mul_vec(W['D_q'][g*16:(g+1)*16], u[g*16:(g+1)*16, t])
            y_ssm[g*16:(g+1)*16, t] = sat_add(y_ch, du)

    y_gated = pe_mul_vec(y_ssm, lut_silu(z_gate))
    out     = pe_mac_mv(y_gated, W['w_out_q'])

    return dict(p1_norm=p1_norm, x_inner=x_inner, z_gate=z_gate,
                x_conv=x_conv, u=u, xproj=xproj, delta=delta,
                y_ssm=y_ssm, y_gated=y_gated, out=out)


def run_mamba_test_hw(p1_q, c, fb=FB):
    """rmsnorm_hw_fast path (infer_mamba_hw logic)."""
    d_in    = p1_q.shape[0]
    T       = p1_q.shape[1]
    d_inner = c['d_inner']
    d_state = c['d_state']
    dt_rank = c['dt_rank']

    p1_norm = rmsnorm_hw_fast(p1_q, c['w_norm_q'], d_in)
    x_inner = pe_mac_mv(p1_norm, c['w_mx_q'])
    z_gate  = pe_mac_mv(p1_norm, c['w_mz_q'])

    x_pad_dw = np.pad(x_inner.astype(np.int64), ((0, 0), (3, 0)))
    idx_dw   = np.arange(4)[None, :] + np.arange(T)[:, None]
    x_unf_dw = x_pad_dw[:, idx_dw]
    x_conv   = np.einsum('dtk,dk->dt', x_unf_dw, c['w_dw_q'].astype(np.int64))
    x_conv   = sat_add(sat16(x_conv >> fb), c['b_dw_q'][:, None])

    u      = lut_silu(x_conv)
    xproj  = pe_mac_mv(u, c['w_xp_q'])

    dt_raw = xproj[:dt_rank, :]
    delta  = sat16((c['w_dt_q'].astype(np.int64) @ dt_raw.astype(np.int64)) >> fb)
    delta  = lut_softplus(sat_add(delta, c['b_dt_q'][:, None]))

    B = xproj[dt_rank:dt_rank+d_state, :]
    C = xproj[dt_rank+d_state:dt_rank+2*d_state, :]
    A = c['A_q']
    D = c['D_q']
    h = np.zeros((d_inner, d_state), np.int64)
    y_ssm = np.zeros((d_inner, T), np.int64)

    for t in range(T):
        dt_t  = delta[:, t:t+1]
        da_in = sat16((dt_t * A.astype(np.int64)) >> fb)
        dA    = lut_exp(da_in)
        dB    = sat16((dt_t * B[:, t].astype(np.int64)[np.newaxis, :]) >> fb)
        t1    = sat16((dA.astype(np.int64) * h) >> fb)
        t2    = sat16((dB.astype(np.int64) * u[:, t:t+1]) >> fb)
        h     = sat_add(t1, t2)
        y_ch  = sat16((h.astype(np.int64) @ C[:, t].astype(np.int64)) >> fb)
        du    = sat16((D.astype(np.int64) * u[:, t].astype(np.int64)) >> fb)
        y_ssm[:, t] = sat_add(y_ch, du)

    y_gated = pe_mul_vec(y_ssm, lut_silu(z_gate))
    out     = pe_mac_mv(y_gated, c['w_out_q'])

    return dict(p1_norm=p1_norm, x_inner=x_inner, z_gate=z_gate,
                x_conv=x_conv, u=u, xproj=xproj, delta=delta,
                y_ssm=y_ssm, y_gated=y_gated, out=out)


def main():
    params = get_config('config.yaml')
    params['exp_type'] = 'super'
    dev = torch.device('cpu')

    batch_size = params['hyperparameters']['batch_size']
    _, _, test_loader, num_class, _ = get_loaders(
        params['data'], params['exp_type'], batch_size=batch_size)

    model = ITMN(n_classes=num_class, **params['model']).to(dev)
    model.load_state_dict(
        torch.load(params['test_ckpt_path'], map_location=dev)['model_state_dict'])
    model.eval()

    _, block_caches = cache_weights(model)

    sample = next(iter(test_loader))
    waveform_t = sample['waveform'][:1].to(dev)
    with torch.no_grad():
        enc_out = model.encoder(waveform_t.transpose(-1, -2))
    x_q = q(enc_out[0].cpu().numpy())

    # Test block 0 only
    c = block_caches[0]
    bk = c['bk']
    blk = model.layers[bk]

    p1_q = hw_p1(x_q, c['w_p1_q'], c['b_p1_q'])

    print('=' * 60)
    print('Mamba stage-by-stage: extract (hw_rms_norm_inline loop) vs test_hw (rmsnorm_hw_fast vectorised)')
    print('=' * 60)

    # Build weight dict for extract path
    mixer   = blk.mamba_block.mixer
    W       = mamba_stages_extract(mixer, p1_q)
    w_norm_q = q(to_f32(blk.mamba_block.norm.weight))

    E = run_mamba_extract(p1_q, w_norm_q, W)
    H = run_mamba_test_hw(p1_q, c)

    for key in ['p1_norm', 'x_inner', 'z_gate', 'x_conv', 'u',
                'xproj', 'delta', 'y_ssm', 'y_gated', 'out']:
        chk(key, E[key], H[key])

    print()
    # Also compare the two rmsnorm functions directly
    print('Direct rmsnorm comparison on p1_q:')
    rms_e = hw_rms_norm_inline(p1_q, w_norm_q, p1_q.shape[0])
    rms_h = rmsnorm_hw_fast(p1_q, c['w_norm_q'], p1_q.shape[0])
    chk('rms_inline vs rms_fast', rms_e, rms_h)

    # Show a few differing samples if any
    diff = np.abs(rms_e.ravel() - rms_h.ravel())
    if diff.max() > 0:
        idx = np.where(diff > 0)[0][:5]
        print(f"\n  First differing indices: {idx}")
        print(f"  extract vals: {rms_e.ravel()[idx]}")
        print(f"  test_hw vals: {rms_h.ravel()[idx]}")
        print(f"  p1_q vals:    {p1_q.ravel()[idx]}")
        print(f"  w_norm vals:  {w_norm_q.ravel()[idx % p1_q.shape[0]]}")


if __name__ == '__main__':
    main()
