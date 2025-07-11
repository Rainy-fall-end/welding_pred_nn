import torch
def build_causal_mask(L: int, device=None):
    """
    返回 shape=(L,L) 的上三角布尔掩码。
    True  表示“禁止(attn_score=-∞)”，False 表示“允许”。
    """
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    return mask            # (L,L)
