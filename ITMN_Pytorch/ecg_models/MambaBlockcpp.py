import torch
import torch.nn as nn
import numpy as np

try:
    import mamba_cpp
except ImportError:
    print("CẢNH BÁO: Không thể import module C++ 'mamba_cpp'.")
    print("Lớp MambaBlockCpp sẽ không hoạt động.")
    mamba_cpp = None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class MambaBlockCpp(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        if mamba_cpp is None:
            raise ImportError("Module C++ 'mamba_cpp' không có sẵn.")

        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        D_INNER = d_model * 2
        DT_RANK = (d_model + 15) // 16
        D_STATE = 16
        D_CONV = 4

        self.mixer = nn.Module()
        self.mixer.A_log = nn.Parameter(torch.empty(D_INNER, D_STATE))
        self.mixer.D = nn.Parameter(torch.empty(D_INNER))
        self.mixer.in_proj = nn.Linear(d_model, 2 * D_INNER, bias=False) 
        self.mixer.conv1d = nn.Conv1d(in_channels=D_INNER, out_channels=D_INNER, 
                                      kernel_size=D_CONV, bias=True, groups=D_INNER, padding=D_CONV - 1)
        self.mixer.x_proj = nn.Linear(D_INNER, DT_RANK + D_STATE * 2, bias=False)
        self.mixer.dt_proj = nn.Linear(DT_RANK, D_INNER, bias=True)
        self.mixer.out_proj = nn.Linear(D_INNER, d_model, bias=False) 

    def forward(self, x):
        x_norm = self.norm(x)
        weights_dict = {
            "rms_norm_weight": self.norm.weight.detach().cpu().numpy(),
            "A_log":           self.mixer.A_log.detach().cpu().numpy(),
            "D":               self.mixer.D.detach().cpu().numpy(),
            "conv1d_weight":   self.mixer.conv1d.weight.detach().cpu().numpy(),
            "conv1d_bias":     self.mixer.conv1d.bias.detach().cpu().numpy(),
            "x_proj_weight":   self.mixer.x_proj.weight.detach().cpu().numpy(),
            "dt_proj_weight":  self.mixer.dt_proj.weight.detach().cpu().numpy(),
            "dt_proj_bias":    self.mixer.dt_proj.bias.detach().cpu().numpy(),
            "out_proj_weight": self.mixer.out_proj.weight.detach().cpu().numpy(),
        }
        in_proj_w_full = self.mixer.in_proj.weight.detach().cpu().numpy()
        in_proj1_w, in_proj2_w = np.split(in_proj_w_full, 2, axis=0)
        weights_dict["in_proj1_weight"] = in_proj1_w
        weights_dict["in_proj2_weight"] = in_proj2_w
        
        batch_size, seq_len, d_model = x.shape
        outputs = []
        for i in range(batch_size):
            input_np = x_norm[i].detach().cpu().numpy()
            output_np = mamba_cpp.forward(input_np, weights_dict)
            outputs.append(torch.from_numpy(output_np))

        final_output = torch.stack(outputs, dim=0).to(x.device)
        
        return final_output