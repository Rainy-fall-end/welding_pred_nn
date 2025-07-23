import torch
import torch.nn as nn
from layers.timesvec import Time2Vec
class ARImputer(nn.Module):
    """
    Autoregressive imputer (方案 A：占位 token)
    ---------------------------------------------
    输入
        x_seq : (B,S,D)  – 已观测帧特征（教师强制）
        para  : (B,2)    – 全局参数
        t_seq : (B,S)    – 每帧 start_time
        p_seq : (B,S)    – 每帧 period
    输出
        y_hat : (B,S-1,D)  – 对 x₁..x_{S-1} 的并行预测
    """
    def __init__(self, d, d_model=256, d_time=16,
                 nhead=8, nlayer=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.D = d
        # ① Token embed: [x_s ; para] → d_model
        self.token_proj = nn.Linear(d + 2, d_model)

        # ② Time embed (Time2Vec 或 sin/cos)
        self.time_s  = Time2Vec(d_time)
        self.time_c  = Time2Vec(d_time)
        self.time_proj = nn.Linear(2 * d_time, d_model)

        # ③ Transformer encoder (带因果 mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayer)

        # ④ Prediction head
        self.head = nn.Linear(d_model, d)

    # ---------- forward：并行 teacher-forcing ----------
    def forward(self, x_raw, para, start_times, time_periods):
        """
        x_raw        : (B,S,D)      – x₀..x_{S-1}
        start_times  : (B,S)        – t₀..t_{S-1}
        time_periods : (B,S)        – p₀..p_{S-1}
        返回
            y_hat   : (B,S-1,D)     – 预测 x₁..x_{S-1}
        """
        B, S, D = x_raw.shape
        device  = x_raw.device

        # ---------- 1) 构造“占位 token” ----------
        pad_x = torch.zeros(B, 1, D, device=device)           # (B,1,D) – dummy x̂_S
        x_in  = torch.cat([x_raw[:, :-1, :], pad_x], dim=1)    # (B,S,D)

        # 时间特征向左移一位，使位置 i 的时间对齐目标 x_{i}
        t_in = torch.cat([start_times[:, 1:],  start_times[:, -1:]],  dim=1)  # (B,S)
        p_in = torch.cat([time_periods[:, 1:], time_periods[:, -1:]], dim=1)  # (B,S)

        # ---------- 2) Embedding ----------
        para_exp = para.unsqueeze(1).expand(-1, S, -1)          # (B,S,2)
        tok_emb  = self.token_proj(torch.cat([x_in, para_exp], -1))  # (B,S,E)

        time_emb = self.time_proj(
            torch.cat([self.time_s(t_in), self.time_c(p_in)], -1))   # (B,S,E)

        h = tok_emb + time_emb                                      # (B,S,E)

        # ---------- 3) Causal mask ----------
        causal = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
        h = self.transformer(h, mask=causal)                        # (B,S,E)

        # ---------- 4) 预测 x₁..x_{S-1} ----------
        y_hat = self.head(h[:, :-1, :])                             # (B,S-1,D)
        return y_hat

    # ---------- generate：逐帧自回归 ----------
    @torch.no_grad()
    def generate(self, x0, para, start_times, time_periods):
        """
        x0   : (B,D)          – 首帧
        start_times, time_periods : (B,S)
        返回
            seq : (B,S,D)      – 包含 x0 和生成的后续帧
        """
        B, D = x0.shape
        S    = start_times.size(1)
        device = x0.device

        # 初始化序列：未知帧填 0
        seq = torch.zeros(B, S, D, device=device)
        seq[:, 0] = x0

        for s in range(1, S):
            # 把当前已生成的帧 + 未来占位 0 一并送入
            y_hat = self.forward(seq, para, start_times, time_periods)  # (B,S-1,D)
            seq[:, s] = y_hat[:, s-1]     # 用 h_{s-1} 预测的值填充第 s 帧
        return seq
