import torch
import torch.nn as nn
from layers.attention import MLP
import math

# ------------------------------------------------------------
# 1. random + 两端必选
# ------------------------------------------------------------
# 1. 随机采样 (含首尾)
def sample_random_with_ends(T: int, k: int, batch_size: int) -> torch.LongTensor:
    """
    随机采样 k 帧，强制包含首尾 (0, T-1)，批量生成。
    返回: (batch_size, k)
    """
    assert 2 <= k <= T, f"T={T}, k={k}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    first, last = 0, T - 1
    mid_pool = torch.arange(1, T - 1, device=device)
    
    idx_list = []
    for _ in range(batch_size):
        rand_idx = mid_pool[torch.randperm(mid_pool.numel())[: k - 2]]
        idx = torch.cat([torch.tensor([first, last], device=device), rand_idx]).sort()[0]
        idx_list.append(idx)
    return torch.stack(idx_list, dim=0)  # (B, k)


# 2. 等步长采样 (含末帧)
def sample_strided(T: int, k: int, batch_size: int) -> torch.LongTensor:
    """
    等步长采样，自动含末帧。
    返回: (batch_size, k)
    """
    assert 2 <= k <= T, f"T={T}, k={k}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stride = math.ceil(T / (k - 1))
    idx_base = torch.arange(0, T, stride, device=device)
    if idx_base.shape[0] > k:
        idx_base = idx_base[:k]
    if idx_base[-1] != T - 1:
        if idx_base.shape[0] == k:
            idx_base[-1] = T - 1
        else:
            idx_base = torch.cat([idx_base, torch.tensor([T - 1], device=device)])
    idx_base = torch.unique(idx_base).sort()[0]

    # 所有batch相同
    return idx_base.unsqueeze(0).expand(batch_size, -1)  # (B, k)


# 3. 全部保留
def sample_keep_all(T: int, batch_size: int) -> torch.LongTensor:
    """
    保留所有帧。
    返回: (batch_size, T)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(0).expand(batch_size, -1)  # (B, T)

# ---------- Gumbel Top‑k -------------------------------------------------
def gumbel_topk(logits: torch.Tensor, k: int, tau: float = 1.0):
    g = -torch.empty_like(logits).exponential_().log()
    y = (logits + g) / tau
    return y.topk(k, dim=-1).indices                     # (B, k)

# ---------- 主模块 -------------------------------------------------------
class GumbelSelector(nn.Module):
    def __init__(self, hidden_dim: int, T: int, k: int = 5, tau: float = 1.0):
        """
        hidden_dim : 统一隐藏维
        T          : 序列长度（含首尾），需 ≥ 3 才能有中间帧
        k          : 采样帧数 (含首尾)，满足 2 ≤ k ≤ T
        tau        : Gumbel‑Softmax 温度
        """
        super().__init__()
        assert 2 <= k <= T, f"k 应在 [2, T]，当前 T={T}, k={k}"
        self.T, self.k, self.tau = T, k, tau

        self.para_emb = MLP(2, hidden_dim * 2, hidden_dim)

        self.feat_proj = nn.LazyLinear(hidden_dim)

        self.emb_table = nn.Parameter(torch.randn(T - 2, hidden_dim) * 0.02)

        self.score_mlp = MLP(4 * hidden_dim, hidden_dim, 1)


    def forward(self, x):
        """
        x = (feat_seq, para_tensor)
        feat_seq    : (B, D_in)          # ✔ 只有特征向量，没有时间维
        para_tensor : (B, 2)
        return      : indices (B, k)  LongTensor
        """
        feat_seq, para = x
        B, D_in = feat_seq.shape
        T       = self.T                 # 仍按照初始化时设定的序列长度

        # ---- 1. Para embedding ------------------------------------------------
        para_emb = self.para_emb(para)                       # (B, h)
        feat_cat = torch.cat([feat_seq, para_emb], dim=-1)   # (B, D_in + h)

        # ---- 2. 线性投影 ------------------------------------------------------
        feat_proj = self.feat_proj(feat_cat)                 # (B, h)

        # ---- 3. Target Attention (DIN) ---------------------------------------
        emb_tbl  = self.emb_table            # (T-2, h) — learnable
        emb_exp  = emb_tbl.unsqueeze(0)      # (1, T-2, h) → broadcast
        emb_exp  = emb_exp.expand(B, -1, -1) # (B, T-2, h)

        q        = feat_proj.unsqueeze(1).expand(-1, T-2, -1)   # (B, T-2, h)

        din_in   = torch.cat([q, emb_exp, q - emb_exp, q * emb_exp], dim=-1)  # (B, T-2, 4h)
        scores   = self.score_mlp(din_in).squeeze(-1)            # (B, T-2)

        # ---- 4. Gumbel‑Top‑k  -------------------------------------------------
        sel_mid  = gumbel_topk(scores, k=self.k - 2, tau=self.tau)  # (B, k-2)

        # ---- 5. 拼首尾索引 & 排序 -------------------------------------------
        idx = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=sel_mid.device),       # 起点 0
            sel_mid + 1,                                                      # 中间帧偏移 +1
            torch.full((B, 1), T - 1, dtype=torch.long, device=sel_mid.device) # 终点 T-1
        ], dim=1)
        idx, _ = torch.sort(idx, dim=1)       # (B, k) 升序

        return idx


