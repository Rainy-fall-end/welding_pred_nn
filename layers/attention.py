import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim:int = None, drop: float = 0.):
        super().__init__()
        if not out_dim:
            out_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x
    
class DWT2D(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.wave = pywt.Wavelet(wave)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        coeffs = []
        for i in range(C):
            ch = x[:, i, :, :].unsqueeze(1)  # (B,1,H,W)
            LL, (LH, HL, HH) = zip(*[pywt.dwt2(ch[b,0].cpu().numpy(), self.wave, mode='periodization') for b in range(B)])
            coeffs.append((
                torch.tensor(LL).to(x.device),
                torch.tensor(LH).to(x.device),
                torch.tensor(HL).to(x.device),
                torch.tensor(HH).to(x.device)
            ))
        LL = torch.stack([c[0] for c in coeffs], dim=1)
        LH = torch.stack([c[1] for c in coeffs], dim=1)
        HL = torch.stack([c[2] for c in coeffs], dim=1)
        HH = torch.stack([c[3] for c in coeffs], dim=1)
        return LL, LH, HL, HH

class IWT2D(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.wave = pywt.Wavelet(wave)

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        rec = []
        for b in range(B):
            chs = []
            for c in range(C):
                coeffs = (LL[b, c].cpu().numpy(),
                          (LH[b, c].cpu().numpy(), HL[b, c].cpu().numpy(), HH[b, c].cpu().numpy()))
                recon = pywt.idwt2(coeffs, self.wave, mode='periodization')
                chs.append(torch.tensor(recon).to(LL.device))
            rec.append(torch.stack(chs, dim=0))  # (C,H,W)
        return torch.stack(rec, dim=0)  # (B,C,H,W)

class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=True, attn_drop=0., proj_drop=0., wave='haar'):
        super().__init__()
        self.dwt = DWT2D(wave)
        self.iwt = IWT2D(wave)
        self.num_heads = num_heads

        # 频率通道变换：从4个子带 → dim
        self.transform = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=qkv_bias),
            nn.ReLU(inplace=True),
            nn.Dropout(attn_drop),
            nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B,C,H,W)

        LL, LH, HL, HH = self.dwt(x)
        out = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4C, H/2, W/2)
        out = self.transform(out)  # (B, C, H/2, W/2)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = out.flatten(2).permute(0, 2, 1)  # (B, N, C)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))
