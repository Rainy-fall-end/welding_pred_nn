import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalWeightedLoss(nn.Module):
    def __init__(self, seq_len: int, main_weight: float = 0.8):
        super().__init__()
        assert 0 < main_weight < 1, "main_weight must be in (0,1)"
        self.seq_len = seq_len
        self.main_weight = main_weight
        self.aux_weight = 1.0 - main_weight
        self.aux_logits = nn.Parameter(torch.zeros(seq_len - 1))  # (T-1,)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred, target: (B, T, C, H, W)
        returns:
            last_frame_loss: (scalar)
            weighted_loss: (scalar)
        """
        B, T, C, H, W = pred.shape
        assert T == self.seq_len, f"Expected seq_len {self.seq_len}, got {T}"

        # MSE loss per frame: (B, T)
        loss_per_frame = F.mse_loss(pred, target, reduction='none').mean(dim=(2, 3, 4))  # (B, T)

        # ---- 1. 最后一帧 loss ----
        last_frame_loss = loss_per_frame[:, -1].mean()

        # ---- 2. 帧权重 ----
        aux_weights = torch.softmax(self.aux_logits, dim=0)  # (T-1,)
        aux_weights = aux_weights * self.aux_weight          # scale to sum = 1 - main_weight
        weights = torch.cat([aux_weights, pred.new_tensor([self.main_weight])], dim=0)  # (T,)

        # ---- 3. 加权 loss ----
        weighted_loss = (loss_per_frame * weights).mean()

        return last_frame_loss, weighted_loss
