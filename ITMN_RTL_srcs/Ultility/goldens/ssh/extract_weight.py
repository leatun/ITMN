import torch
import numpy as np
import os
import argparse
import yaml

# Import các thành phần cần thiết
from ecg_models.ITMN import ITMN
from dataset import get_loaders
from utils.utils import get_config

# --- CẤU HÌNH ---
OUTPUT_DIR = 'golden_vectors'
TARGET_LAYER_NAME = 'layers.0.mamba_block' 

def extract(params, ckpt_path):
    # --- 1. SETUP MÔ HÌNH VÀ DỮ LIỆU ---
    print("--- 1. Đang setup mô hình và dữ liệu... ---")
    model_config = params['model']
    exp_type = params['exp_type']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Đang sử dụng device: {device}")

    _, _, test_loader, num_class, _ = get_loaders(params['data'], exp_type, batch_size=1)
    real_sample = next(iter(test_loader))
    real_waveform = real_sample['waveform']
    print(f"   - Đã lấy 1 mẫu dữ liệu thật từ test set, shape: {real_waveform.shape}")

    model = ITMN(n_classes=num_class, **model_config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   - Nạp mô hình từ checkpoint '{ckpt_path}' thành công.")

    # --- 2. ĐĂNG KÝ HOOK ---
    captured_io = {}
    def get_activation_hook(name):
        def hook(model, input, output):
            # Hook đã có .cpu() nên nó đã an toàn
            captured_io[name + '_input'] = input[0].detach().cpu().numpy()
            captured_io[name + '_output'] = output.detach().cpu().numpy()
        return hook

    target_module = dict(model.named_modules()).get(TARGET_LAYER_NAME)
    if target_module is None:
        raise ValueError(f"Không tìm thấy lớp có tên '{TARGET_LAYER_NAME}'")
    
    target_module.register_forward_hook(get_activation_hook(TARGET_LAYER_NAME))
    print(f"--- 2. Đã đăng ký hook thành công vào lớp: '{TARGET_LAYER_NAME}' ---")

    # --- 3. CHẠY FORWARD PASS ---
    print("--- 3. Đang chạy forward pass để trích xuất dữ liệu... ---")
    with torch.no_grad():
        model(real_waveform.to(device))
    print("   - Chạy forward pass hoàn tất.")

    # --- 4. LƯU GOLDEN INPUT/OUTPUT ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 4. Đang lưu golden vectors vào thư mục '{OUTPUT_DIR}'... ---")
    
    golden_input = captured_io[TARGET_LAYER_NAME + '_input']
    golden_output = captured_io[TARGET_LAYER_NAME + '_output']

    golden_input[0].astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'golden_input.bin'))
    golden_output[0].astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'golden_output.bin'))
    print(f"   - Đã lưu 'golden_input.bin' (Shape: {golden_input[0].shape})")
    print(f"   - Đã lưu 'golden_output.bin' (Shape: {golden_output[0].shape})")

    # --- 5. LƯU TRỌNG SỐ CỦA LỚP MỤC TIÊU (ĐÃ SỬA) ---
    print("--- 5. Đang lưu trọng số của lớp mục tiêu... ---")
    with torch.no_grad():
        # Trọng số của RMSNorm
        target_module.norm.weight.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'rms_norm_weight.bin'))

        # Trọng số của Mamba mixer
        mixer = target_module.mixer
        
        # in_proj.weight (không có bias)
        in_proj_w, in_proj2_w = np.split(mixer.in_proj.weight.cpu().numpy(), 2, axis=0)
        in_proj_w.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'in_proj1_weight.bin'))
        in_proj2_w.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'in_proj2_weight.bin'))

        # conv1d (có bias)
        mixer.conv1d.weight.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'conv1d_weight.bin'))
        mixer.conv1d.bias.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'conv1d_bias.bin'))
        
        # x_proj (không có bias)
        mixer.x_proj.weight.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'x_proj_weight.bin'))
        
        # dt_proj (có bias)
        mixer.dt_proj.weight.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'dt_proj_weight.bin'))
        mixer.dt_proj.bias.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'dt_proj_bias.bin'))
        
        # A_log và D
        mixer.A_log.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'A_log.bin'))
        mixer.D.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'D.bin'))
        
        # out_proj (không có bias)
        mixer.out_proj.weight.cpu().numpy().astype(np.float32).tofile(os.path.join(OUTPUT_DIR, 'out_proj_weight.bin'))
    print("   - Lưu trọng số hoàn tất.")
    print(f"\n✅ TẤT CẢ CÁC FILE ĐÃ ĐƯỢC LƯU VÀO THƯ MỤC '{OUTPUT_DIR}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='super', help='Experiment type')
    args = parser.parse_args()
    
    config = get_config('config.yaml')
    config['exp_type'] = args.exp_type.lower()
    
    ckpt_path = config['test_ckpt_path']
    extract(config, ckpt_path)