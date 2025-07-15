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

def sample_frames_with_ends(x: torch.Tensor, k: int):
    """
    从时间维度 T 中随机采样 k 个时间步，强制保留第一个和最后一个时间帧
    输入:
        x: Tensor, shape (B, T, C, H, W)
        k: int, 要采样的时间帧数, 必须 >= 2
    返回:
        x_sampled: Tensor, shape (B, k, C, H, W)
        selected_indices: List[int], 被采样的时间帧索引
    """
    B, T, C, H, W = x.shape
    assert k >= 2, "k must be at least 2"
    assert T >= k, f"T={T} is less than k={k}"

    # 固定帧索引：起始帧和结束帧
    first_idx = 0
    last_idx = T - 1

    # 可采样区间：[1, T-2]
    candidate_indices = list(range(1, T - 1))
    num_random = k - 2

    # 随机采样不重复时间帧
    random_indices = torch.randperm(len(candidate_indices))[:num_random]
    selected_middle = [candidate_indices[i.item()] for i in random_indices]

    # 合并并排序索引
    selected_indices = [first_idx] + selected_middle + [last_idx]
    selected_indices.sort()  # 可选：保持时间顺序

    # 采样张量
    x_sampled = x[:, selected_indices]  # shape: (B, k, C, H, W)

    return x_sampled, selected_indices
