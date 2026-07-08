import torch
import numpy as np
import os
import argparse
import yaml # Cần cài đặt: pip install pyyaml

# Import các thành phần cần thiết từ dự án của bạn
from ecg_models.ITMN import ITMN

# --- HELPER FUNCTIONS (Mô phỏng lại logic của bạn) ---

def get_config(config_path):
    """Nạp file config YAML."""
    with open(config_path, 'r') as stream:
        return yaml.safe_load(stream)

def get_num_classes(exp_type):
    """
    Trả về số lượng class tương ứng với mỗi exp_type.
    Đây là phiên bản đơn giản hóa của get_loaders, chỉ lấy số class.
    """
    # Các giá trị này được lấy từ paper và cấu trúc dataset PTB-XL
    class_map = {
        'super': 5,
        'sub': 23,
        'rhythm': 12,
        'all': 71,
        'diag': 44,
        'form': 19,
        'cpsc': 9
    }
    if exp_type not in class_map:
        raise ValueError(f"exp_type '{exp_type}' không hợp lệ.")
    return class_map[exp_type]

# --- SCRIPT CHÍNH ---

if __name__ == '__main__':
    # --- 1. PARSE ARGUMENTS (Giống hệt test.py) ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type', type=str, default='super', 
                        choices=['super', 'sub', 'rhythm', 'all', 'diag', 'form', 'cpsc'],
                        help='Loại thí nghiệm để xác định số class và checkpoint.')
    args = parser.parse_args()

    # --- 2. NẠP CẤU HÌNH ĐỘNG ---
    print(f"--- Đang nạp cấu hình cho exp_type: '{args.exp_type}' ---")
    config = get_config('config.yaml')
    
    # Lấy các thông số từ file config
    model_config = config['model']
    ckpt_path = config['test_ckpt_path']
    
    # Xác định số class động dựa trên exp_type
    num_class = get_num_classes(args.exp_type)

    print(f"   - d_model: {model_config.get('d_model', 'Không có')}")
    print(f"   - Số class: {num_class}")
    print(f"   - Checkpoint: '{ckpt_path}'")

    if not os.path.exists(ckpt_path):
        print(f"\nLỖI: Không tìm thấy file checkpoint tại '{ckpt_path}'")
        exit()

    # --- 3. KHỞI TẠO VÀ NẠP MODEL (Giống hệt test.py) ---
    print("\n--- Đang khởi tạo và nạp mô hình... ---")
    try:
        # Khởi tạo model với các tham số động
        model = ITMN(n_classes=num_class, **model_config)
        
        # Nạp checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # Nạp state_dict từ key 'model_state_dict'
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Nạp mô hình thành công.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi khi nạp mô hình hoặc checkpoint: {e}")
        print("Hãy đảm bảo các tham số trong config.yaml khớp với file checkpoint.")
        exit()

    # --- 4. PHÂN TÍCH TRỌNG SỐ ---
    # Chọn khối Mamba bạn muốn phân tích
    TARGET_BLOCK_NAME = 'layers.0.mamba_block' 
    
    target_block = dict(model.named_modules()).get(TARGET_BLOCK_NAME)

    if target_block is None:
        print(f"\nLỖI: Không tìm thấy module có tên '{TARGET_BLOCK_NAME}' trong mô hình.")
        exit()

    print(f"\n--- Phân Tích Trọng Số Cho Khối: '{TARGET_BLOCK_NAME}' ---")
    for param_name, param_tensor in target_block.named_parameters():
        numpy_tensor = param_tensor.detach().cpu().numpy()
        
        print("\n" + "="*60)
        print(f"Tên Parameter (trong PyTorch): {param_name}")
        print(f"Shape (Kích thước): {param_tensor.shape}")
        print(f"Tổng số phần tử: {param_tensor.numel()}")
    
    print("\n" + "="*60)
    print("\n✅ Phân tích hoàn tất.")