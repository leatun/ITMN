import torch
import numpy as np
import os
import argparse

from ecg_models.ITMN import ITMN
from dataset import get_loaders
from utils.utils import get_config

# --- CAU HINH ---
OUTPUT_DIR = 'golden_vectors'
TARGET_LAYER_NAME = 'layers.0.mamba_block'


def save_and_report(arr, filename):
    """Luu file .bin va in shape + dtype"""
    path = os.path.join(OUTPUT_DIR, filename)
    arr = arr.astype(np.float32)
    arr.tofile(path)
    print(f"   - Saved {filename:25s} | shape={arr.shape} | dtype={arr.dtype}")


def extract(params, ckpt_path):
    print("\n=== SETUP MODEL & DATA ===")
    model_config = params['model']
    exp_type = params['exp_type']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    _, _, test_loader, num_class, _ = get_loaders(
        params['data'], exp_type, batch_size=1
    )

    real_sample = next(iter(test_loader))
    real_waveform = real_sample['waveform']
    print(f"Input waveform shape: {real_waveform.shape}")

    model = ITMN(n_classes=num_class, **model_config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Checkpoint loaded: {ckpt_path}")

    # --- HOOK ---
    captured_io = {}

    def hook_fn(name):
        def hook(_, inp, out):
            captured_io[name + "_input"] = inp[0].detach().cpu().numpy()
            captured_io[name + "_output"] = out.detach().cpu().numpy()
        return hook

    target_module = dict(model.named_modules()).get(TARGET_LAYER_NAME)
    if target_module is None:
        raise ValueError(f"Layer not found: {TARGET_LAYER_NAME}")

    target_module.register_forward_hook(hook_fn(TARGET_LAYER_NAME))
    print(f"Hook registered on: {TARGET_LAYER_NAME}")

    # --- FORWARD ---
    with torch.no_grad():
        model(real_waveform.to(device))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== SAVE GOLDEN INPUT / OUTPUT ===")
    save_and_report(captured_io[TARGET_LAYER_NAME + "_input"][0], "golden_input.bin")
    save_and_report(captured_io[TARGET_LAYER_NAME + "_output"][0], "golden_output.bin")

    print("\n=== SAVE LAYER WEIGHTS ===")
    with torch.no_grad():
        # RMSNorm
        save_and_report(
            target_module.norm.weight.cpu().numpy(),
            "rms_norm_weight.bin"
        )

        mixer = target_module.mixer

        # in_proj (split)
        in_proj1, in_proj2 = np.split(
            mixer.in_proj.weight.cpu().numpy(), 2, axis=0
        )
        save_and_report(in_proj1, "in_proj1_weight.bin")
        save_and_report(in_proj2, "in_proj2_weight.bin")

        # conv1d
        save_and_report(mixer.conv1d.weight.cpu().numpy(), "conv1d_weight.bin")
        save_and_report(mixer.conv1d.bias.cpu().numpy(), "conv1d_bias.bin")

        # x_proj
        save_and_report(mixer.x_proj.weight.cpu().numpy(), "x_proj_weight.bin")

        # dt_proj
        save_and_report(mixer.dt_proj.weight.cpu().numpy(), "dt_proj_weight.bin")
        save_and_report(mixer.dt_proj.bias.cpu().numpy(), "dt_proj_bias.bin")

        # A_log, D
        save_and_report(mixer.A_log.cpu().numpy(), "A_log.bin")
        save_and_report(mixer.D.cpu().numpy(), "D.bin")

        # out_proj
        save_and_report(mixer.out_proj.weight.cpu().numpy(), "out_proj_weight.bin")

    print(f"\nDONE. Tat ca file da duoc luu trong '{OUTPUT_DIR}'\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='super')
    args = parser.parse_args()

    config = get_config('config.yaml')
    config['exp_type'] = args.exp_type.lower()

    extract(config, config['test_ckpt_path'])
