import torch
import torch.nn as nn
from typing import Tuple

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

# --- 模型主体 ------------------------------------------------
class ARImputer(nn.Module):
    """
    输入
    ----
    x_raw  : (B, D)        – 当前帧特征
    para   : (B, 2)        – 全局 2-D 参数
    t_all  : (B, S)        – S 个 start_time
    p_all  : (B, S)        – S 个 period

    输出
    ----
    y_pred : (B, S-1, D)   – 从 t₁ 到 t_{S-1} 的预测
    """
    def __init__(self,
                 d: int,                 # D
                 d_model: int = 256,
                 d_para: int = 32,
                 d_time: int = 16,
                 nhead: int = 8,
                 num_layers: int = 2,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # ----- 1. Memory token (key/value) -----
        self.mem_proj = nn.Linear(d + 2, d_model)     # x_raw + para -> mem

        # ----- 2. Query token embedding -----
        self.time_s  = Time2Vec(d_time)
        self.time_c  = Time2Vec(d_time)
        self.q_proj  = nn.Linear(2 * d_time, d_model)

        # ----- 3. Cross-Attention stack -----
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   d_ff, dropout,
                                                   batch_first=True)
        self.xattn = nn.TransformerEncoder(encoder_layer,
                                           num_layers=num_layers)

        # ----- 4. Prediction head -----
        self.head = nn.Linear(d_model, d)

    # ----------------------------------------------------------
    def forward(self,
                x_raw: torch.Tensor,          # (B,D)
                para: torch.Tensor,           # (B,2)
                start_times: torch.Tensor,    # (B,S)
                time_periods: torch.Tensor) -> torch.Tensor:  # (B,S)

        B, D = x_raw.shape
        _, S = start_times.shape
        assert S > 1, "S 必须 ≥2，否则输出为空"

        # ===== 1) Memory token =====
        mem_token = self.mem_proj(torch.cat([x_raw, para], dim=-1))  # (B,d_model)
        mem_token = mem_token.unsqueeze(1)                           # (B,1,E)

        # ===== 2) Query tokens (含 t0, …, t_{S-1}) =====
        t_feat = self.time_s(start_times)         # (B,S,d_time)
        p_feat = self.time_c(time_periods)        # (B,S,d_time)
        q_emb  = self.q_proj(torch.cat([t_feat, p_feat], dim=-1))    # (B,S,E)

        # ===== 3) 拼接形成序列：1 mem + S query =====
        #    位置：[ mem | q0 | q1 | ... | q_{S-1} ]
        tokens = torch.cat([mem_token, q_emb], dim=1)                # (B, 1+S, E)

        # ===== 4) Cross-Attention Encoder =====
        #    令 self-Attn 能够让每个 query 看到 mem_token
        enc_out = self.xattn(tokens)                                 # (B,1+S,E)

        # ===== 5) 取第 2..S+1 位 (= q1..q_{S-1}) 预测 =====
        q_out  = enc_out[:, 2:, :]                # (B,S-1,E)
        y_pred = self.head(q_out)                 # (B,S-1,D)

        return y_pred
