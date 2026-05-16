import torch
import torch.nn as nn
import torch.nn.functional as F
from ecg_models.MambaBlockcpp import MambaBlockCpp
from mamba_ssm import Mamba


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super(MambaBlock, self).__init__()
        self.mixer = Mamba(d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.mixer(self.norm(x))

        return x


class BaseInceptionBlock(nn.Module):
    # Biến cờ để chỉ in 1 lần duy nhất, tránh trôi màn hình
    _printed_shapes = True 

    def __init__(self, d_model):
        super(BaseInceptionBlock, self).__init__()
        dim = d_model // 4
        self.bottleneck = nn.Conv1d(d_model, dim, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv1d(dim, dim, kernel_size=39, stride=1, padding=19, bias=False)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=19, stride=1, padding=9, bias=False)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=9, stride=1, padding=4, bias=False)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv1d(d_model, dim, kernel_size=1, stride=1, bias=False)

        self.bn = nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not BaseInceptionBlock._printed_shapes:
            print("\n" + "="*65)
            print("🔍 TRACE SHAPE BÊN TRONG BASE_INCEPTION_BLOCK ĐẦU TIÊN")
            print("="*65)
            print(f"0. Input (x) ban đầu        : {x.shape} --> (Batch, d_model, Length)")

        output = self.bottleneck(x)
        if not BaseInceptionBlock._printed_shapes:
            print(f"1. Sau Bottleneck (k=1)     : {output.shape} --> Ép xuống {output.shape[1]} channels")

        output4 = self.conv4(output)
        output3 = self.conv3(output)
        output2 = self.conv2(output)
        if not BaseInceptionBlock._printed_shapes:
            print(f"2a. Nhánh Conv4 (k=39)      : {output4.shape}")
            print(f"2b. Nhánh Conv3 (k=19)      : {output3.shape}")
            print(f"2c. Nhánh Conv2 (k=9)       : {output2.shape}")

        output1 = self.maxpool(x)
        output1 = self.conv1(output1)
        if not BaseInceptionBlock._printed_shapes:
            print(f"3. Nhánh MaxPool -> Conv1   : {output1.shape}")

        cat_out = torch.cat((output1, output2, output3, output4), dim=1)
        if not BaseInceptionBlock._printed_shapes:
            print(f"4. Sau Concatenate (Gộp 4)  : {cat_out.shape} --> Phục hồi lại {cat_out.shape[1]} channels")

        x_out = self.relu(self.bn(cat_out))
        if not BaseInceptionBlock._printed_shapes:
            print(f"5. Output cuối cùng (x_out) : {x_out.shape}")
            print("="*65 + "\n")
            BaseInceptionBlock._printed_shapes = True # Đánh dấu đã in xong

        return x_out


class ISSMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ISSMBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        self.inception_block = BaseInceptionBlock(out_channels)
        self.mamba_block = MambaBlock(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.inception_block(x)
        x2 = self.relu(self.mamba_block(x.transpose(-1, -2)).transpose(-1, -2))
        x = x1 + x2

        return x


class ITMBlock(nn.Module): #MambaBlock
    def __init__(self, in_channels, out_channels, use_cpp_mamba = False):
        super(ITMBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        self.inception_block = BaseInceptionBlock(out_channels)
        if use_cpp_mamba:
            print(f"   -> ITMBlock({in_channels}, {out_channels}) đang sử dụng MambaBlockCpp.")
            self.mamba_block = MambaBlockCpp(out_channels)
        else:
            self.mamba_block = MambaBlock(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.inception_block(x)
        x2 = self.relu(self.mamba_block(x.transpose(-1, -2)).transpose(-1, -2))
        x = x1 + x2

        return x


class ITMN(nn.Module):
    def __init__(self, d_model, n_classes=2, use_cpp_mamba=False):
        """Full Mamba model."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(12, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
        )
        self.layers = nn.Sequential(
            ITMBlock(d_model, d_model, use_cpp_mamba=use_cpp_mamba),
            ITMBlock(d_model, d_model, use_cpp_mamba=use_cpp_mamba),
            nn.MaxPool1d(2, 2),
            ITMBlock(d_model, d_model, use_cpp_mamba=False),
            ITMBlock(d_model, d_model, use_cpp_mamba=False),
            nn.MaxPool1d(2, 2),
            ITMBlock(d_model, 2 * d_model, use_cpp_mamba=False),
        )

        self.classifier = nn.Linear(2 * d_model, n_classes)

        self.apply(init_weights)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = self.layers(x)
        x = x.mean(dim=-1)

        x = self.classifier(x)

        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


if __name__ == '__main__':
    device = 'cuda'
    x = torch.rand((4, 1000, 12)).to(device)
    model = ITMN(d_model=64, n_classes=5).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    out = model(x)
    print(out.shape)
