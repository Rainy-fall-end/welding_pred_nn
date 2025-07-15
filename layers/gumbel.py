import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, k=5, tau=1.0):
        super().__init__()
        self.k = k        # number of tokens to select
        self.tau = tau    # temperature for Gumbel-Softmax

        # 网络用于产生 selection logits（每个 token 的重要性）
        self.selector_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个 score
        )

    def forward(self, x):
        """
        x: 输入序列, shape = [B, T, D]
        返回: 选中的 token 组合, shape = [B, k, D]
        """
        B, T, D = x.shape
        
        # [B, T, 1] -> 得到每个 token 的重要性 logits
        logits = self.selector_net(x).squeeze(-1)  # shape: [B, T]

        # Gumbel-Top-k 采样：添加 Gumbel noise
        gumbel_noise = -torch.empty_like(logits).exponential_().log()
        y = logits + gumbel_noise
        topk = torch.topk(y, self.k, dim=-1)[1]  # 选出 top-k 的索引

        # 构造 one-hot mask 并乘上输入，得到选择后的子序列
        mask = torch.zeros_like(logits).scatter_(1, topk, 1.0)
        mask = mask.unsqueeze(-1)  # shape: [B, T, 1]
        
        # Soft mask version (可以换成硬 mask)
        x_selected = x * mask

        # 提取非零 token
        selected_tokens = []
        for i in range(B):
            selected = x_selected[i][mask[i].squeeze(-1)[i] > 0]
            selected_tokens.append(selected)
        # 若需要 [B, k, D]，可使用以下简单版本：
        selected = torch.gather(x, 1, topk.unsqueeze(-1).expand(-1, -1, D))
        
        return selected  # shape: [B, k, D]
    
if __name__ == "__main__":
    # 用法示例
    B, T, D = 2, 10, 32  # batch, sequence length, embedding dim
    x = torch.randn(B, T, D)

    selector = GumbelSelector(input_dim=D, hidden_dim=64, k=3, tau=0.5)
    selected = selector(x)  # shape: [B, 3, D]

    print("Selected shape:", selected.shape)
