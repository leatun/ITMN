# File: profile_model_internals.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

# Import các thành phần cần thiết từ chính dự án ITMN
from dataset import get_loaders
from ecg_models.ITMN import ITMN
from utils.utils import get_config
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

# ===================================================================
# 1. THIẾT LẬP GỠ LỖI
# ===================================================================
DEBUG_DIR = "cpp_golden_files"
os.makedirs(DEBUG_DIR, exist_ok=True)
STEP_COUNTER = 0

def save_tensor(tensor, name):
    """Hàm tiện ích để lưu tensor trung gian với chỉ số tự động tăng."""
    global STEP_COUNTER
    filename = f"{STEP_COUNTER:02d}_{name}.txt"
    full_path = os.path.join(DEBUG_DIR, filename)
    
    tensor_to_save = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor
    tensor_np = tensor_to_save.detach().cpu().numpy()
    
    print(f"DEBUG: Đang lưu {filename} (shape: {tensor_np.shape})")
    
    if tensor_np.ndim > 2:
        tensor_np = tensor_np.reshape(-1, tensor_np.shape[-1])
        
    np.savetxt(full_path, tensor_np, fmt='%.8e')
    STEP_COUNTER += 1

# ===================================================================
# 2. SCRIPT CHÍNH
# ===================================================================
if __name__ == '__main__':
    # --- Lấy cấu hình từ dòng lệnh và file config.yaml (GIỐNG HỆT test.py) ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='all', help='Loại tác vụ (all, diag, ...)')
    args = parser.parse_args()

    config = get_config('config.yaml')
    config['exp_type'] = args.exp_type.lower()
    model_config = config['model']
    hyperparameters = config['hyperparameters']
    ckpt_path = config['test_ckpt_path']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # --- Tải dữ liệu và mô hình theo đúng cách của test.py ---
    print(f"\nĐang tải dữ liệu cho tác vụ '{args.exp_type}'...")
    _, _, test_loader, num_class, _ = get_loaders(config['data'], args.exp_type, 1)

    print(f"\nĐang tải mô hình và checkpoint từ '{ckpt_path}'...")
    model = ITMN(n_classes=num_class, **model_config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Tải thành công!")

    # --- Lấy một mẫu duy nhất từ test_loader ---
    print("\nLấy mẫu đầu tiên từ tập test...")
    first_sample = next(iter(test_loader))
    waveform = first_sample['waveform'][0:1].to(device, dtype=torch.float)
    SEQ_LEN = waveform.shape[1]

    # --- Cô lập các thành phần cần thiết ---
    encoder = model.encoder
    itm_block_0 = model.layers[0]
    mamba_block = itm_block_0.mamba_block
    mixer = mamba_block.mixer
    
    # Lấy các hằng số từ mixer
    D_INNER = mixer.d_inner
    D_STATE = mixer.d_state
    DT_RANK = mixer.dt_rank

    print(f"\n--- Bắt đầu Profile MambaBlock trên một mẫu dữ liệu (SEQ_LEN={SEQ_LEN}) ---")

    with torch.no_grad():
        # --- Bắt đầu luồng forward pass ---
        
        # --- ITMN.forward ---
        save_tensor(waveform, "00_ITMN_input_waveform")
        x = waveform.transpose(-1, -2)
        save_tensor(x, "01_ITMN_transposed")
        x = encoder(x)
        save_tensor(x, "02_ITMN_after_encoder")

        # --- ITMBlock.forward ---
        itm_block_input = x
        save_tensor(itm_block_input, "03_ITMBlock_input")
        x_conv_itm = itm_block_0.conv(itm_block_input)
        save_tensor(x_conv_itm, "04_ITMBlock_after_conv")
        
        # --- Inception Branch ---
        x1_inception = itm_block_0.inception_block(x_conv_itm)
        save_tensor(x1_inception, "05_ITMBlock_inception_branch_out")
        
        # --- Mamba Branch ---
        mamba_block_input = x_conv_itm.transpose(-1, -2)
        save_tensor(mamba_block_input, "06_MambaBlock_input")
        
        # --- MambaBlock.forward ---
        x_norm = mamba_block.norm(mamba_block_input)
        save_tensor(x_norm, "07_MambaBlock_after_norm")
        
        # --- Mamba (mixer).forward ---
        xz = mixer.in_proj(x_norm)
        save_tensor(xz, "X_after_linear")
        x_mixer, z_mixer = xz.chunk(2, dim=-1)
        
        x_mixer_transposed = x_mixer.transpose(1, 2)
        x_conv = mixer.conv1d(x_mixer_transposed)
        x_conv_sliced = x_conv[..., :SEQ_LEN]
        x_activated = F.silu(x_conv_sliced)
        save_tensor(x_activated, "08_Mixer_x_activated")
        
        x_act_rearranged = x_activated.transpose(1, 2)
        x_dbl = mixer.x_proj(x_act_rearranged)
        dt_raw, B_raw, C_raw = torch.split(x_dbl, [DT_RANK, D_STATE, D_STATE], dim=-1)
        
        delta = F.softplus(mixer.dt_proj(dt_raw)).transpose(1, 2)
        save_tensor(delta, "09_Mixer_delta_final")
        save_tensor(B_raw, "10_Mixer_B_raw")
        save_tensor(C_raw, "11_Mixer_C_raw")
        
        A = -torch.exp(mixer.A_log.float())
        B_transposed = B_raw.transpose(1, 2)
        C_transposed = C_raw.transpose(1, 2)
        
        # Scan thủ công để lấy h
        discrete_A = torch.exp(A.unsqueeze(0).unsqueeze(2) * delta.unsqueeze(3))
        discrete_B = delta.unsqueeze(3) * B_raw.unsqueeze(1)
        deltaB_u = discrete_B * x_activated.unsqueeze(3)
        
        ssm_state = torch.zeros(x_activated.shape[0], D_INNER, D_STATE, device=device, dtype=torch.float32)
        for i in range(SEQ_LEN):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            save_tensor(ssm_state, f"12_Mixer_h_state_t{i}")
        
        # Chạy lại scan để lấy output
        ssm_state.zero_()
        scan_outputs = []
        for i in range(SEQ_LEN):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output_i = torch.matmul(ssm_state, C_raw[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output_i.squeeze(-1))
        scan_output_raw = torch.stack(scan_outputs, dim=-1)
        save_tensor(scan_output_raw, "13_Mixer_scan_output_raw")
        
        # Gating và out_proj
        y_gated = scan_output_raw * F.silu(z_mixer.transpose(1, 2))
        save_tensor(y_gated, "14_Mixer_y_gated")
        
        mixer_output = mixer.out_proj(y_gated.transpose(1, 2))
        save_tensor(mixer_output, "15_Mixer_final_output")
        
        # --- Quay lại ITMBlock.forward ---
        x2_mamba = itm_block_0.relu(mixer_output).transpose(-1, -2)
        save_tensor(x2_mamba, "16_ITMBlock_mamba_branch_out_final")
        
        x_sum = x1_inception + x2_mamba
        save_tensor(x_sum, "17_ITMBlock_final_output")

    print(f"\nĐã tạo thành công tất cả file profile trong thư mục '{DEBUG_DIR}'")