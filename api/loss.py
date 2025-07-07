import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalWeightedLoss(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        # 初始化时让末帧权重大一些
        init_w = torch.ones(seq_len)
        init_w[-1] = seq_len            # e.g.  T 倍
        self.weight = nn.Parameter(init_w)

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target, reduction='none') \
                 .mean(dim=(2,3,4))                      # (B,T)
        norm_w = torch.softmax(self.weight, dim=0)       # 归一化
        return (loss * norm_w).mean()
