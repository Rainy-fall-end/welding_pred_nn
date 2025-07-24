import torch
import torch.nn.functional as F
import numpy as np
def build_causal_mask(L: int, device=None):
    """
    返回 shape=(L,L) 的上三角布尔掩码。
    True  表示“禁止(attn_score=-∞)”，False 表示“允许”。
    """
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    return mask            # (L,L)

def compute_variable_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    对 10 个通道（3 位移, 6 应力, 1 温度）分别计算：
        1) 最后一个时间步的 MAE/MSE/RMSE/R²
        2) 整个序列 (T) 的平均 MAE/MSE/RMSE/R²
    输入:
        pred, target: (B, T, C=10, H, W)
    输出:
        dict ──{
            displacement: {last: {...}, overall: {...}},
            stress      : {last: {...}, overall: {...}},
            temperature : {last: {...}, overall: {...}}
        }
    """
    B, T, C, H, W = pred.shape
    assert C == 10, "Expected 10 channels: 3 disp, 6 stress, 1 temp"

    # ---------- 拆分 ----------
    disp_pred_last    = pred[:, -1, 0:3]      # (B, 3, H, W)
    stress_pred_last  = pred[:, -1, 3:9]
    temp_pred_last    = pred[:, -1, 9:10]

    disp_pred_all     = pred[:, :, 0:3]       # (B, T, 3, H, W)
    stress_pred_all   = pred[:, :, 3:9]
    temp_pred_all     = pred[:, :, 9:10]

    disp_target_last  = target[:, -1, 0:3]
    stress_target_last= target[:, -1, 3:9]
    temp_target_last  = target[:, -1, 9:10]

    disp_target_all   = target[:, :, 0:3]
    stress_target_all = target[:, :, 3:9]
    temp_target_all   = target[:, :, 9:10]

    # ---------- 统一计算 ----------
    def compute_metrics(x: torch.Tensor, y: torch.Tensor):
        """
        x, y: 任意维度相同的 Tensor
        return: dict of scalar errors
        """
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        mae  = F.l1_loss(x_flat, y_flat, reduction='mean')
        mse  = F.mse_loss(x_flat, y_flat, reduction='mean')
        rmse = torch.sqrt(mse)
        var  = torch.var(y_flat, unbiased=False)
        r2   = 1 - mse / (var + 1e-8)

        return {
            "mae":  mae.item(),
            "mse":  mse.item(),
            "rmse": rmse.item(),
            "r2":   r2.item()
        }

    # ---------- 输出 ----------
    return {
        "displacement": {
            "last":    compute_metrics(disp_pred_last,   disp_target_last),
            "overall": compute_metrics(disp_pred_all,    disp_target_all)
        },
        "stress": {
            "last":    compute_metrics(stress_pred_last, stress_target_last),
            "overall": compute_metrics(stress_pred_all,  stress_target_all)
        },
        "temperature": {
            "last":    compute_metrics(temp_pred_last,   temp_target_last),
            "overall": compute_metrics(temp_pred_all,    temp_target_all)
        }
    }

def gather_by_idx(x, idx):
    # x : (B, T, C, H, W, ...)
    # idx: (B, k)
    B, k = idx.shape
    # 先把 idx 的后续维度补 1
    idx_exp = idx.view(B, k, *([1] * (x.dim() - 2)))
    # 再扩展到 x 的剩余维度
    idx_exp = idx_exp.expand(-1, -1, *x.shape[2:])  # (B, k, C, H, W, ...)
    return torch.gather(x, 1, idx_exp)               # (B, k, C, H, W, ...)

def flatten_dict(d, parent_key="", sep="/"):
    """
    将任意层级嵌套字典展开为扁平字典，键名通过 sep 拼接
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # 如果值是 numpy / torch.Tensor，可转 float
            try:
                if isinstance(v, torch.Tensor):
                    v = v.item()
                elif isinstance(v, np.generic):
                    v = v.item()
            except ImportError:
                pass
            items.append((new_key, v))
    return dict(items)

def split_params(model,keywords="gumbel_selector"):
    gumbel_params = []
    other_params = []
    for name, p in model.named_parameters():
        if keywords in name:
            gumbel_params.append(p)
        else:
            other_params.append(p)
    return gumbel_params, other_params

def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return None
    return sum(vals) / len(vals)

def flatten_with_slash(d, parent_key=""):
    flat = {}
    for k, v in d.items():
        nk = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_with_slash(v, nk))
        else:
            # to python scalar
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            flat[nk] = v
    return flat
