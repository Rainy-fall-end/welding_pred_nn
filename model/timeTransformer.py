import torch, torch.nn as nn, math

# ---------- 1) Time2Vec ----------
# Paper: Kazemi et al., “Time2Vec: Learning a Vector Representation of Time”, 2019
# https://arxiv.org/abs/1907.05321
class Time2Vec(nn.Module):
    def __init__(self, k):                        # k = d_time
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))    # 线性分量
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(k-1))
        self.b  = nn.Parameter(torch.randn(k-1))

    def forward(self, t):                         # t:(B,S) float (start 或 con)
        # 线性 + 各频率正弦
        lin = (self.w0 * t + self.b0).unsqueeze(-1)          # (B,S,1)
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)   # (B,S,k-1)
        return torch.cat([lin, per], dim=-1)                 # (B,S,k)

# ---------- 2) relative_time_bias ----------
# Idea inspired by Shaw et al., “Self-Attention with Relative Position Representations”, ACL 2018
def relative_time_bias(start):
    # start:(B,S)  ->  Δt:(B,S,S)
    dt = (start.unsqueeze(2) - start.unsqueeze(1)).abs()     # 秒或分钟
    # 任意可学习函数；此处简单指数衰减
    return (-dt / 3600.0).exp()                              # (B,S,S)

# ---------- 3) 主模块 ----------
class IrregularImputer(nn.Module):
    def __init__(self, d, d_model=256, d_time=16,
                 nhead=8, nlayers=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.in_proj  = nn.Linear(d, d_model - d_time)
        self.t2v_s    = Time2Vec(d_time)   # start
        self.t2v_c    = Time2Vec(d_time)   # duration
        self.mask_tok = nn.Parameter(torch.zeros(d_model))
        self.pos_emb  = nn.Parameter(torch.zeros(1, 1000, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, nlayers)
        self.out_proj = nn.Linear(d_model, d)

    def forward(self, x, mask, start, con):
        B, S, _ = x.shape
        # ---- 1) 内容 + 时间投影 ----
        x_val  = self.in_proj(x)                              # (B,S,d_model-d_time)
        x_time = torch.cat([self.t2v_s(start), self.t2v_c(con)], dim=-1)
        x_all  = torch.cat([x_val, x_time], dim=-1)           # (B,S,d_model)

        # 替换缺失位置为 mask_token
        x_all = torch.where(mask.unsqueeze(-1), self.mask_tok, x_all)
        x_all = x_all + self.pos_emb[:, :S]

        # ---- 2) 自定义注意力偏置 ----
        bias = relative_time_bias(start)                      # (B,S,S)
        # nn.TransformerEncoder 不原生支持 bias；可改写 MultiheadAttention：
        attn_mask = None
        def add_bias(module):
            for name, mod in module.named_children():
                if isinstance(mod, nn.TransformerEncoderLayer):
                    mha = mod.self_attn
                    mha.register_forward_pre_hook(
                        lambda _m, input: (input[0], input[1],
                                           input[2] + bias))  # 偏置加到 attn_mask
        add_bias(self.encoder)

        h = self.encoder(x_all, attn_mask=attn_mask)          # (B,S,d_model)
        y_pred = self.out_proj(h)

        y = x.clone()
        y[mask] = y_pred[mask]
        return y
