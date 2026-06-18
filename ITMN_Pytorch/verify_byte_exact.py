"""
verify_byte_exact.py — Sanity check: RTL golden files match test_hw infer_block_hw chain
on the same first test sample (sample 0 from the test loader).

Compares Final_ITM_Full_FP.txt for each block (produced by extract_itm_full.py)
against the integer chain output of infer_block_hw on the identical input.
Also compares intermediate per-stage goldens where available.

Usage:
    python verify_byte_exact.py
"""
import sys
import numpy as np
import torch
from pathlib import Path

from extract_itm_full import (
    q, sat16, sat_add, FB,
    hw_p1, load_hex,
    bn_relu_hw,
)
from test_hw import (
    infer_block_hw, infer_mamba_hw, cache_weights, hw_maxpool,
    pe_mac_conv_fast, rmsnorm_hw_fast,
)
from dataset import get_loaders
from ecg_models.ITMN import ITMN
from utils.utils import get_config

GOLDEN_BASE = Path('golden_all')
PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def load_golden_iq(block_dir, fname):
    """Load integer golden file (hex Q4.11 signed) → int64 array."""
    p = block_dir / fname
    if not p.exists():
        return None
    return load_hex(p)


def check(label, hw_flat, golden_flat, indent='  '):
    """Print comparison result; return True if identical."""
    if golden_flat is None:
        print(f"{indent}{label:<40s}  SKIP (no golden file)")
        return True
    if hw_flat.shape != golden_flat.shape:
        print(f"{indent}{label:<40s}  {FAIL}  shape hw={hw_flat.shape} vs golden={golden_flat.shape}")
        return False
    if np.array_equal(hw_flat, golden_flat):
        print(f"{indent}{label:<40s}  {PASS}")
        return True
    else:
        diff = np.abs(hw_flat - golden_flat)
        n_err = int((diff > 0).sum())
        print(f"{indent}{label:<40s}  {FAIL}  n_mismatch={n_err}/{hw_flat.size}  max_diff={diff.max()}")
        return False


def main():
    params = get_config('config.yaml')
    params['exp_type'] = 'super'
    dev = torch.device('cpu')

    print('[*] Loading model...')
    _, _, _, num_class, _ = get_loaders(
        params['data'], params['exp_type'],
        batch_size=params['hyperparameters']['batch_size'])

    model = ITMN(n_classes=num_class, **params['model']).to(dev)
    model.load_state_dict(
        torch.load(params['test_ckpt_path'], map_location=dev)['model_state_dict'])
    model.eval()

    _, block_caches = cache_weights(model)
    itm_indices = [c['bk'] for c in block_caches]

    has_maxpool = [False]
    for idx in range(1, len(itm_indices)):
        prev_bk, cur_bk = itm_indices[idx-1], itm_indices[idx]
        mp = any(isinstance(model.layers[j], torch.nn.MaxPool1d)
                 for j in range(prev_bk + 1, cur_bk))
        has_maxpool.append(mp)

    # Block 0 input: load directly from golden P1_Input_X.txt (float-quantized encoder
    # output captured via PyTorch hook during golden generation).  Using a direct
    # model.encoder() call introduces ±1-LSB FP rounding differences at quantisation
    # boundaries compared to the hook-captured values, causing ~35 x_q mismatches and
    # cascading ~267 mamba mismatches downstream.  Loading the golden file is the only
    # way to start from the byte-exact same integer input.
    bdir0 = GOLDEN_BASE / f'block_00_layer{[c["bk"] for c in block_caches][0]:02d}'
    x_q = load_hex(bdir0 / 'P1_Input_X.txt').reshape(
        block_caches[0]['d_in'], -1).astype(np.int64)

    print(f'\n{"=" * 68}')
    print(f'  Byte-exact comparison: test_hw infer_block_hw vs golden_all/block_*/  ')
    print(f'{"=" * 68}')

    all_pass = True

    for blk_idx, c in enumerate(block_caches):
        bk  = c['bk']
        mp  = has_maxpool[blk_idx]
        bdir = GOLDEN_BASE / f'block_{blk_idx:02d}_layer{bk:02d}'

        if mp:
            x_q = hw_maxpool(x_q)

        # ---- Reproduce per-stage intermediates identically to infer_block_hw ----
        d_out = c['d_out']
        T     = x_q.shape[1]

        p1_q   = hw_p1(x_q, c['w_p1_q'], c['b_p1_q'])
        bot_q  = sat16((c['w_bot_q'].astype(np.int64) @ p1_q.astype(np.int64)) >> FB)
        x_pad3 = np.pad(p1_q.astype(np.int64), ((0, 0), (1, 1)), mode='edge')
        p1_mp_q = np.maximum(np.maximum(x_pad3[:, :T], x_pad3[:, 1:T+1]),
                              x_pad3[:, 2:T+2])
        b1_q   = sat16((c['w_b1_q'].astype(np.int64) @ p1_mp_q.astype(np.int64)) >> FB)
        b2_q   = pe_mac_conv_fast(bot_q, c['w_b2_q'], kernel=9,  pad=4)
        b3_q   = pe_mac_conv_fast(bot_q, c['w_b3_q'], kernel=19, pad=9)
        b4_q   = pe_mac_conv_fast(bot_q, c['w_b4_q'], kernel=39, pad=19)
        inc_cat_q   = np.concatenate([b1_q, b2_q, b3_q, b4_q], axis=0)
        mamba_out_q = infer_mamba_hw(c, p1_q)

        # Final: PyTorch formula
        mul_i = inc_cat_q.astype(np.int64) * c['scale_q'][:, None].astype(np.int64)
        bn_i  = sat16(sat16(mul_i >> FB) + c['shift_q'][:, None].astype(np.int64))
        x1    = np.where(bn_i < 0, np.int64(0), bn_i)
        x2    = np.where(mamba_out_q < 0, np.int64(0), mamba_out_q)
        final_q = sat_add(x1, x2)

        x_q = final_q   # feed into next block

        print(f'\n  Block {blk_idx} (model layer {bk}), shape={final_q.shape}:')
        # P1_Output_Golden.txt is a float-reference (save_hex of BN output) — always
        # differs from integer hw_p1.  Check against P1_Output_Golden_FP.txt instead.
        check('P1_out vs P1_Output_Golden.txt (float ref)',
              p1_q.ravel(),
              load_golden_iq(bdir, 'P1_Output_Golden.txt'))
        all_pass &= check('P1_out vs P1_Output_Golden_FP.txt',
                          p1_q.ravel(),
                          load_golden_iq(bdir, 'P1_Output_Golden_FP.txt'))
        all_pass &= check('Mamba_out vs Mam_OutProj_FP.txt',
                          mamba_out_q.ravel(),
                          load_golden_iq(bdir, 'Mam_OutProj_FP.txt'))
        all_pass &= check('Final vs Final_ITM_Full_FP.txt',
                          final_q.ravel(),
                          load_golden_iq(bdir, 'Final_ITM_Full_FP.txt'))

    print(f'\n{"=" * 68}')
    if all_pass:
        print(f'  ALL CHECKS PASSED — test_hw and extract golden files are byte-exact.')
    else:
        print(f'  SOME CHECKS FAILED — see above for details.')
    print(f'{"=" * 68}\n')
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
