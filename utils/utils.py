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

