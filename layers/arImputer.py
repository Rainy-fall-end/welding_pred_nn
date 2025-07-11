import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
# --- 时间嵌入（沿用你之前的 Time2Vec） ---------------------
class Time2Vec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(k - 1))
        self.b  = nn.Parameter(torch.randn(k - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,)  – 秒数或其他连续时间标量
        lin = (self.w0 * t + self.b0).unsqueeze(-1)    # (...,1)
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)  # (...,k-1)
        return torch.cat([lin, per], dim=-1)                 # (...,k)

class ARImputer(nn.Module):
    """
    LLM-style autoregressive predictor for continuous frames.
    ---------------------------------------------
    输入
        x_seq : (B,S,D)  – 帧特征序列 x₀..x_{S-1}
        para  : (B,2)    – 全局参数
        t_seq : (B,S)    – start_time
        p_seq : (B,S)    – period
    输出
        y_hat : (B,S-1,D)  – 对 x₁..x_{S-1} 的并行预测
    """
    def __init__(self, D, d_model=256, d_time=16,
                 nhead=8, nlayer=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.D = D
        # ① Token embed: [x_s ; para] → d_model
        self.token_proj = nn.Linear(D+2, d_model)

        # ② Time embed  (Time2Vec or sin/cos)
        self.time_s = Time2Vec(d_time)
        self.time_c = Time2Vec(d_time)
        self.time_proj = nn.Linear(2*d_time, d_model)

        # ③ Transformer decoder (仅需 encoder+causal mask)
        block = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(block, nlayer)

        # ④ Prediction head
        self.head = nn.Linear(d_model, D)

    # ---------- forward：并行 teacher-forcing ----------
    def forward(self, x_raw, para, start_times, time_periods):
        B, S, D = x_raw.shape
        para_exp = para.unsqueeze(1).expand(-1, S, -1)         # (B,S,2)
        tok_in   = torch.cat([x_raw, para_exp], -1)            # (B,S,D+2)
        tok_emb  = self.token_proj(tok_in)                     # (B,S,E)

        # time embedding
        ts_emb = self.time_proj(
            torch.cat([self.time_s(start_times), self.time_c(time_periods)], -1))  # (B,S,E)

        h = tok_emb + ts_emb                                   # (B,S,E)

        # causal mask (float: 0 or -inf)
        causal = torch.triu(torch.ones(S, S, device=x_raw.device) * float('-inf'),
                            diagonal=1)
        h = self.transformer(h, mask=causal)                   # (B,S,E)

        y_hat = self.head(h[:, :-1, :])                        # (B,S-1,D)
        return y_hat           # 预测 x₁..x_{S-1}

    # ---------- generate：逐帧自回归 ----------
    @torch.no_grad()
    def generate(self, x0, para, start_times, time_periods):
        """
        x0   : (B,D)      – 首帧
        t_all, p_all : (B,S)
        返回 y_pred : (B,S,D) ，含 x0
        """
        B, D = x0.shape
        S    = start_times.size(1)
        device = x0.device

        seq  = torch.zeros(B, S, D, device=device)
        seq[:, 0] = x0

        for s in range(1, S):
            y_hat = self.forward(seq[:, :s, :],                # 历史
                                 para,
                                 start_times[:, :s], time_periods[:, :s])   # 对应时间
            x_next = y_hat[:, -1, :]                           # 取最后一步预测
            seq[:, s] = x_next
        return seq
