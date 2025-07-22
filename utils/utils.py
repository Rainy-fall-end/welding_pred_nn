import torch
import torch.nn.functional as F
def build_causal_mask(L: int, device=None):
    """
    返回 shape=(L,L) 的上三角布尔掩码。
    True  表示“禁止(attn_score=-∞)”，False 表示“允许”。
    """
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    return mask            # (L,L)

def compute_variable_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    计算最后一个时间步，各变量的 MAE, MSE, RMSE, R²
    输入:
        pred, target: (B, T, C=10, H, W)
    输出:
        dict: 包含每类变量的四个误差指标
    """
    B, T, C, H, W = pred.shape
    assert C == 10, "Expected 10 channels: 3 disp, 6 stress, 1 temp"

    pred_last = pred[:, -1]     # (B, C, H, W)
    target_last = target[:, -1]

    # 变量切分
    disp_pred = pred_last[:, 0:3]
    stress_pred = pred_last[:, 3:9]
    temp_pred = pred_last[:, 9:10]

    disp_target = target_last[:, 0:3]
    stress_target = target_last[:, 3:9]
    temp_target = target_last[:, 9:10]

    def compute_metrics(x: torch.Tensor, y: torch.Tensor):
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        mae = F.l1_loss(x_flat, y_flat, reduction='mean')
        mse = F.mse_loss(x_flat, y_flat, reduction='mean')
        rmse = torch.sqrt(mse)
        var_y = torch.var(y_flat, unbiased=False)
        r2 = 1 - mse / (var_y + 1e-8)
        return {
            "mae": mae.item(),
            "mse": mse.item(),
            "rmse": rmse.item(),
            "r2": r2.item()
        }

    return {
        "displacement": compute_metrics(disp_pred, disp_target),
        "stress": compute_metrics(stress_pred, stress_target),
        "temperature": compute_metrics(temp_pred, temp_target)
    }

import torch
from typing import Tuple, List

def sample_frames_with_ends(
    tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    随机从时间维度采样 k 个时间步，并强制包含首尾帧。
    
    返回:
        out_tensor_sampled   (B, k, C, H, W)
        start_times_sampled  (B, k)
        time_periods_sampled (B, k)
        para_tensor          (B, 2)     —— 原样返回
        selected_indices     list[int]  —— 采样的时间索引
    """
    (out_tensor, start_times, time_periods, para_tensor) = tensor
    B, T, C, H, W = out_tensor.shape
    assert k >= 2, "k must be at least 2"
    assert T >= k, f"T = {T} < k = {k}"

    # ---- 1. 生成索引 --------------------------------------------------------
    first_idx, last_idx = 0, T - 1
    candidate_indices   = list(range(1, T - 1))
    num_random          = k - 2

    rand_idx = torch.randperm(len(candidate_indices))[:num_random]
    mid_idx  = [candidate_indices[i.item()] for i in rand_idx]

    selected_indices = sorted([first_idx] + mid_idx + [last_idx])  # 升序

    # ---- 2. 采样 -----------------------------------------------------------
    out_sample  = out_tensor[:, selected_indices]          # (B, k, C, H, W)
    st_sample   = start_times[:, selected_indices]         # (B, k)
    tp_sample   = time_periods[:, selected_indices]        # (B, k)
    out = (out_sample, st_sample, tp_sample, para_tensor)
    return out, selected_indices

