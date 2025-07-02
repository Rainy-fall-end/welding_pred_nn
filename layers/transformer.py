"""
Autoregressive Imputer (AR-Transformer) for irregular time-series
================================================================
核心组件
--------
1. Time2Vec            – Kazemi et al., 2019
2. Relative-Position   – Shaw et al., ACL 2018 (simplified bias)
3. GPT-style stack     – Vaswani et al., 2017, causal mask only
4. Scheduled sampling  – Bengio et al., 2015  (optional)

Author  : your-name
Created : 2025-07-02
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# 1) Time2Vec –– absolute and duration embeddings
# ---------------------------------------------------------------------------
class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time (Kazemi et al., 2019)
    t: (B, S) float   – seconds since epoch or duration in seconds
    return: (B, S, k)
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(k - 1))
        self.b  = nn.Parameter(torch.randn(k - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        lin = (self.w0 * t + self.b0).unsqueeze(-1)            # (B,S,1)
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)     # (B,S,k-1)
        return torch.cat([lin, per], dim=-1)                   # (B,S,k)


# ---------------------------------------------------------------------------
# 2) Relative position bias (simplified)
# ---------------------------------------------------------------------------
class RelPosBias(nn.Module):
    """
    Shaw et al. 2018 – relative position representations.
    Here we use exponential decay bias f(Δt) = exp(-|Δt| / τ) with learnable τ.
    """
    def __init__(self, num_heads: int, init_tau_hours: float = 1.0):
        super().__init__()
        # one shared scalar τ per head (could be vector)
        tau = torch.ones(num_heads) * (init_tau_hours * 3600.0)
        self.tau = nn.Parameter(tau)

    def forward(self, start: torch.Tensor) -> torch.Tensor:
        """
        start: (B, S) float – absolute start time (seconds)
        return: (B, num_heads, S, S)  bias matrix
        """
        B, S = start.shape
        delta = (start.unsqueeze(2) - start.unsqueeze(1)).abs()      # (B,S,S)
        # (B,1,S,S) then broadcast to heads
        bias = torch.exp(-delta.unsqueeze(1) / self.tau.view(1, -1, 1, 1))
        # log-domain bias (additive to attention scores)
        return torch.log(bias + 1e-9)                                # avoid -inf


# ---------------------------------------------------------------------------
# 3) GPT-style Causal Transformer Block
# ---------------------------------------------------------------------------
class CausalTransformer(nn.Module):
    """
    Stacked TransformerEncoderLayers + causal + relative bias.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 d_ff: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B,S,d_model)
        attn_bias: (B, nhead, S, S) or None
        """
        B, S, _ = x.shape
        # causal mask (True == mask) – upper triangular without diag
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool,
                                       device=x.device), diagonal=1)
        # make it (S,S) → broadcast inside MultiheadAttention
        for layer in self.layers:
            # MultiheadAttention in PyTorch 2.3 supports attn_mask + bias_kq
            # easiest: combine bias in forward pre-hook
            if attn_bias is not None:
                def _pre_hook(mod, inp):
                    q, k, v, need_weights, attn_mask = inp
                    # PyTorch expects (B*nHeads,S,S)
                    bias = attn_bias.reshape(-1, attn_bias.shape[-2],
                                             attn_bias.shape[-1])
                    return (q, k, v, need_weights, attn_mask + bias)
                handle = layer.self_attn.register_forward_pre_hook(_pre_hook)
            x = layer(x, src_mask=causal)
            if attn_bias is not None:
                handle.remove()
        return x


# ---------------------------------------------------------------------------
# 4) AR Imputer as LightningModule
# ---------------------------------------------------------------------------
class ARImputer(pl.LightningModule):
    """
    Inputs
    ------
    x_raw   : (B,S,d)         – raw sequence, zeros where missing
    m_obs   : (B,S) bool      – True if value observed
    start   : (B,S) float     – absolute start timestamp (sec)
    con     : (B,S) float     – duration (sec)

    Outputs
    -------
    y_hat   : (B,S,d)         – network prediction (all steps)
    Loss    : MSE on missing positions
    """
    def __init__(self, d: int,
                 d_model: int = 256,
                 d_time: int = 16,
                 nhead: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 lr: float = 3e-4,
                 sched_sampling_p: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.token_proj = nn.Linear(d + 1, d_model)       # +1 for miss_ind
        self.time_s     = Time2Vec(d_time)
        self.time_c     = Time2Vec(d_time)
        self.pos_bias   = RelPosBias(nhead)
        self.backbone   = CausalTransformer(d_model, nhead, num_layers,
                                            d_ff, dropout)
        self.head       = nn.Linear(d_model, d)
        self.sched_p    = sched_sampling_p

    # --------------------------------------------------------------------- #
    # Helper: create input embedding conditioning on scheduled sampling
    def _prepare_inputs(self, x_true, y_prev, m_obs, step):
        """
        scheduled sampling: with prob p use previous prediction instead of gt
        """
        if self.training and self.sched_p > 0.0 and step > 0:
            prob = self.sched_p
            use_pred = torch.rand_like(m_obs.float()) < prob
            x = torch.where(use_pred.unsqueeze(-1), y_prev, x_true)
            m = torch.where(use_pred, torch.zeros_like(m_obs), m_obs)
        else:
            x = x_true
            m = m_obs
        miss_flag = (~m).float().unsqueeze(-1)
        return torch.cat([x, miss_flag], dim=-1)

    # --------------------------------------------------------------------- #
    def forward(self, x_raw, m_obs, start, con):
        """
        One full forward pass (teacher forcing).
        Returns y_hat for all steps.
        """
        B, S, _ = x_raw.shape
        miss_flag = (~m_obs).float().unsqueeze(-1)
        tok_in = self.token_proj(torch.cat([x_raw, miss_flag], dim=-1))

        # add time embeddings
        t_embed = torch.cat([self.time_s(start), self.time_c(con)], dim=-1)
        h = tok_in + t_embed

        # relative bias for every batch
        bias = self.pos_bias(start)   # (B,nHead,S,S)
        h = self.backbone(h, attn_bias=bias)
        return self.head(h)           # (B,S,d)

    # --------------------------------------------------------------------- #
    # iterative AR generation for inference
    @torch.no_grad()
    def impute(self, x_raw, m_obs, start, con):
        """
        Sequentially fill missing positions left→right using own predictions.
        """
        B, S, d = x_raw.shape
        y = x_raw.clone()
        for t in range(S):
            if ( ~m_obs[:, t] ).any():               # 有缺失
                y_hat = self.forward(y, m_obs, start, con)
                y[:, t, :] = y_hat[:, t, :]
                m_obs[:, t] = True
        return y

    # --------------------------------------------------------------------- #
    # Lightning plumbing
    def training_step(self, batch, batch_idx):
        x_raw, m_obs, start, con = batch
        y_hat = self.forward(x_raw, m_obs, start, con)
        loss = F.mse_loss(y_hat[~m_obs], x_raw[~m_obs])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_raw, m_obs, start, con = batch
        y_hat = self.forward(x_raw, m_obs, start, con)
        loss = F.mse_loss(y_hat[~m_obs], x_raw[~m_obs])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ---------------------------------------------------------------------------
# 5) Tiny synthetic dataset for quick sanity-check
# ---------------------------------------------------------------------------
class ToyDataset(torch.utils.data.Dataset):
    """
    Generate sinusoids + noise; randomly drop 20 % values.
    Each sample: seq_len=50, feature_dim=1
    """
    def __init__(self, N=1024, seq_len=50, d=1):
        super().__init__()
        self.seq_len, self.d = seq_len, d
        t0 = torch.linspace(0, 24*3600, seq_len)            # one day, seconds
        self.start = t0.unsqueeze(0).repeat(N, 1)           # (N,S)
        self.con   = torch.full_like(self.start, 30*60.0)   # 30-min step
        y = torch.sin(2*math.pi * t0 / (24*3600)).unsqueeze(-1)  # (S,1)
        y = y.repeat(N, 1, d)                               # (N,S,d)
        y += 0.05 * torch.randn_like(y)
        self.data_full = y
        mask = torch.rand(N, seq_len) < 0.2
        self.m_obs = ~mask
        y_miss = y.clone()
        y_miss[mask.unsqueeze(-1)] = 0.0                    # drop → 0
        self.data_miss = y_miss

    def __len__(self):
        return self.data_full.shape[0]

    def __getitem__(self, idx):
        return (self.data_miss[idx],
                self.m_obs[idx],
                self.start[idx],
                self.con[idx])


# ---------------------------------------------------------------------------
# 6) Train quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    train_ds = ToyDataset(1024)
    val_ds   = ToyDataset(256)
    loader_tr = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    loader_va = torch.utils.data.DataLoader(val_ds, batch_size=32)

    model = ARImputer(d=1, sched_sampling_p=0.2)
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, loader_tr, loader_va)

    # quick inference on one batch
    x_raw, m_obs, start, con = next(iter(loader_va))
    y_imp = model.impute(x_raw.clone(), m_obs.clone(), start, con)
    print("RMSE (missing positions):",
          torch.sqrt(F.mse_loss(y_imp[~m_obs], x_raw[~m_obs])).item())
