import torch
import torch.nn as nn
# --- 时间嵌入（沿用你之前的 Time2Vec） ---------------------
class Time2Vec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(k - 1))
        self.b  = nn.Parameter(torch.randn(k - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,)  – 秒数或其他连续时间标量
        lin = (self.w0 * t + self.b0).unsqueeze(-1)    # (...,1)
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)  # (...,k-1)
        return torch.cat([lin, per], dim=-1)                 # (...,k)
