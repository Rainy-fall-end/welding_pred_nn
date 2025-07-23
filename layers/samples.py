import torch
import torch.nn as nn
from layers.attention import MLP
import math

# ------------------------------------------------------------
# 1. random + 两端必选
# ------------------------------------------------------------
def sample_random_with_ends(T: int, k: int,*_ignored) -> torch.LongTensor:
    """
    随机采样 k 帧，强制包含首尾 (0, T-1)。

    T : 序列长度
    k : 采样帧数 (>=2 且 <=T)
    """
    assert 2 <= k <= T, f"T={T}, k={k}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    first, last = 0, T - 1
    mid_pool = torch.arange(1, T - 1, device=device)
    rand_idx = mid_pool[torch.randperm(mid_pool.numel())[: k - 2]]
    idx = torch.cat([torch.tensor([first, last], device=device), rand_idx]).sort()[0]
    return idx                                 # (k,)

# ------------------------------------------------------------
# 2. strided 采样 (自动含末帧)
# ------------------------------------------------------------

def sample_strided(T: int, k: int, *_ignored) -> torch.LongTensor:
    assert 2 <= k <= T, f"T={T}, k={k}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 自动推算步长
    stride = math.ceil(T / (k - 1))
    idx = torch.arange(0, T, stride, device=device)

    # 可能多于k个，裁剪
    if idx.shape[0] > k:
        idx = idx[:k]

    # 确保末帧包含
    if idx[-1] != T - 1:
        if idx.shape[0] == k:  # 替换最后一个为末帧
            idx[-1] = T - 1
        else:                  # 追加末帧
            idx = torch.cat([idx, torch.tensor([T - 1], device=device)])

    # 再次排序
    idx = torch.unique(idx).sort()[0]
    return idx

# ------------------------------------------------------------
# 3. keep 全部
# ------------------------------------------------------------
def sample_keep_all(T: int, *_ignored) -> torch.LongTensor:
    """
    不做采样，返回 [0, 1, …, T-1]。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.arange(T, device=device)      # (T,)

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
        T          : 序列长度
        k          : 采样帧数 (含首尾)  k ≥ 2
        tau        : Gumbel 温度
        """
        super().__init__()
        assert 2 <= k <= T, f"k 应在 [2, T], 当前 T={T}, k={k}"
        self.T, self.k, self.tau = T, k, tau

        # 1️⃣  para_tensor (B,2) 嵌入
        self.para_emb = MLP(2, hidden_dim * 2, hidden_dim)

        # 2️⃣  投影 feat_cat → hidden_dim
        self.feat_proj = nn.LazyLinear(hidden_dim)

        # 3️⃣  中间帧可学习 embedding
        self.emb_table = nn.Parameter(torch.randn(T - 2, hidden_dim) * 0.02)

        # 4️⃣  打分 MLP：输入维 = 4 * hidden_dim + 1(dot)
        in_score = hidden_dim * 4 + 1
        self.score_mlp = MLP(in_score, hidden_dim, 1)

    def forward(self, x):
        """
        x = (feat_seq, para_tensor)
        feat_seq    : (B, T, D_in)
        para_tensor : (B, 2)
        return      : indices (B, k)  LongTensor
        """
        feat_seq, para = x
        B, T, _ = feat_seq.shape
        assert T == self.T, f"输入长度 {T} ≠ 初始化 T={self.T}"

        # ---- 1. 拼接 para 嵌入 -------------------------------------------
        para_emb = self.para_emb(para)                     # (B, hidden_dim)
        feat_cat = torch.cat(
            [feat_seq, para_emb.unsqueeze(1).expand(-1, T, -1)], dim=-1
        )                                                  # (B, T, D_in+h)

        # ---- 2. 投影到 hidden_dim ---------------------------------------
        feat_proj = self.feat_proj(feat_cat)               # (B, T, hidden_dim)

        # ---- 3. 取中间帧 & 计算复合特征 -------------------------------
        mid_feat = feat_proj[:, 1:-1, :]                   # (B, T-2, h)
        emb      = self.emb_table.unsqueeze(0)             # (1, T-2, h)

        dot      = (mid_feat * emb).sum(-1, keepdim=True)  # (B, T-2, 1)
        diff     = mid_feat - emb                          # (B, T-2, h)
        prod     = mid_feat * emb                          # (B, T-2, h)

        feat_mix = torch.cat([mid_feat, emb, diff, prod, dot], dim=-1)
        scores   = self.score_mlp(feat_mix).squeeze(-1)    # (B, T-2)

        # ---- 4. Gumbel‑Top‑k 选 k-2 ------------------------------------
        sel_mid = gumbel_topk(scores, k=self.k - 2, tau=self.tau)  # (B, k-2)

        # ---- 5. 拼首尾 & 升序 -----------------------------------------
        idx = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=sel_mid.device),
            sel_mid + 1,                                   # 映射回原序列索引
            torch.full((B, 1), T - 1, dtype=torch.long, device=sel_mid.device)
        ], dim=1)
        idx, _ = torch.sort(idx, dim=1)                    # (B, k)

        return idx

