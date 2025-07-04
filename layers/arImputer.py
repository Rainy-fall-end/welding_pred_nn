# -*- coding: utf-8 -*-
"""
AR-Imputer v2  (trainable impute_all)
====================================
This version removes the `@torch.no_grad()` decorator on `impute_all` so the
looped autoregressive fill **remains differentiable**, enabling training or
fine‑tuning with back‑prop through the whole imputation process.

Inputs
------
    out_tensor           : (B, S, D)
    start_times_tensor   : (B, S)
    time_periods_tensor  : (B, S)
    para_tensor          : (B, 2)
    mask                 : (B, S) bool  – True ⇢ value missing (affects out_tensor only)

Forward behaviour
-----------------
`forward` – fills **the first masked step** for every sample (left‑to‑right).
`impute_all` – autoregressively calls `forward` until no mask remains; now it
runs **with gradients** to support training.

References (paper citations in code):
    • Time2Vec            – Kazemi et al., 2019
    • Relative Position   – Shaw et al., ACL 2018
    • Causal Transformer  – Vaswani et al., 2017
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Time2Vec –– Kazemi et al., 2019
###############################################################################
class Time2Vec(nn.Module):
    """Time2Vec: Learning a Vector Representation of Time (2019)"""
    def __init__(self, k: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))  # linear part
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(k - 1))
        self.b  = nn.Parameter(torch.randn(k - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        lin = (self.w0 * t + self.b0).unsqueeze(-1)                # (B,S,1)
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)         # (B,S,k-1)
        return torch.cat([lin, per], dim=-1)                       # (B,S,k)

###############################################################################
# Relative-time bias –– Shaw et al., ACL 2018 (log-exp variant)
###############################################################################
class RelPosBias(nn.Module):
    """Additive log-bias ~ −|Δt|/τ for each attention head"""
    def __init__(self, num_heads: int, init_tau_hours: float = 1.0):
        super().__init__()
        tau = torch.full((num_heads,), init_tau_hours * 3600.0)     # seconds
        self.tau = nn.Parameter(tau)

    def forward(self, start_times: torch.Tensor) -> torch.Tensor:
        # start_times: (B,S)
        B, S = start_times.shape
        delta = (start_times.unsqueeze(2) - start_times.unsqueeze(1)).abs()  # (B,S,S)
        bias  = torch.exp(-delta.unsqueeze(1) / (self.tau.view(1, -1, 1, 1)))  # (B,H,S,S)
        return torch.log(bias + 1e-9)  # log-bias, add to attention scores

###############################################################################
# Causal Transformer encoder stack (GPT-style)
###############################################################################
class CausalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        layer_fn = lambda: nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True)
        self.layers = nn.ModuleList([layer_fn() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward through N causal layers with optional additive bias."""
        B, S, _ = x.shape
        causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), 1)

        for lyr in self.layers:
            handle = None
            if attn_bias is not None:
                def _pre_hook(mod, inp):
                    q, k, v, need_weights, attn_mask = inp
                    bias = attn_bias.reshape(-1, attn_bias.shape[-2], attn_bias.shape[-1])
                    return (q, k, v, need_weights, attn_mask + bias)
                handle = lyr.self_attn.register_forward_pre_hook(_pre_hook)

            x = lyr(x, src_mask=causal_mask)
            if handle is not None:
                handle.remove()
        return x

###############################################################################
# AR-Imputer v2 (trainable impute_all)
###############################################################################
class ARImputer(nn.Module):
    """Causal Transformer imputer – predicts one missing value per forward call."""

    def __init__(self,
                 d: int,                 # feature dimension of out_tensor
                 d_para: int = 16,       # embedding size for 2‑D global para
                 d_model: int = 256,
                 d_time: int = 16,
                 nhead: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        # projections
        self.para_proj  = nn.Linear(2, d_para)
        self.token_proj = nn.Linear(d + 1 + d_para, d_model)  # +1 miss flag

        # temporal encodings
        self.time_s   = Time2Vec(d_time)
        self.time_c   = Time2Vec(d_time)
        self.pos_bias = RelPosBias(nhead)

        # backbone & head
        self.backbone = CausalTransformer(d_model, nhead, num_layers, d_ff, dropout)
        self.head     = nn.Linear(d_model, d)

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    def _build_input(self, out_tensor: torch.Tensor, mask: torch.Tensor,
                     para_tensor: torch.Tensor) -> torch.Tensor:
        """Create token embeddings with global para + miss‑flag."""
        B, S, _ = out_tensor.shape
        miss_flag = mask.float().unsqueeze(-1)                # (B,S,1)
        para_emb  = self.para_proj(para_tensor).unsqueeze(1)  # (B,1,d_para)
        para_emb  = para_emb.expand(-1, S, -1)                # (B,S,d_para)
        return torch.cat([out_tensor, miss_flag, para_emb], dim=-1)

    # ------------------------------------------------------------------
    def forward(self,
                out_tensor: torch.Tensor,
                start_times_tensor: torch.Tensor,
                time_periods_tensor: torch.Tensor,
                para_tensor: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fill the **first** masked position per sample and return (new_out, new_mask)."""
        B, S, _ = out_tensor.shape

        # token + time embedding
        tok_in = self.token_proj(self._build_input(out_tensor, mask, para_tensor))
        t_embed = torch.cat([self.time_s(start_times_tensor),
                             self.time_c(time_periods_tensor)], dim=-1)
        h = tok_in + t_embed

        # Transformer with relative‑time bias
        bias = self.pos_bias(start_times_tensor)
        h = self.backbone(h, attn_bias=bias)
        y_pred = self.head(h)  # (B,S,D)

        # fill first masked pos per sample
        new_out, new_mask = out_tensor.clone(), mask.clone()
        for b in range(B):
            idxs = torch.nonzero(mask[b], as_tuple=False)
            if idxs.numel() == 0:
                continue
            t = idxs[0].item()
            new_out[b, t] = y_pred[b, t]
            new_mask[b, t] = False
        return new_out, new_mask

    # ------------------------------------------------------------------
    def impute_all(self,
                   out_tensor: torch.Tensor,
                   start_times_tensor: torch.Tensor,
                   time_periods_tensor: torch.Tensor,
                   para_tensor: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """Autoregressively fill **all** masked positions (gradient‑friendly)."""
        new_out, new_mask = out_tensor.clone(), mask.clone()
        while new_mask.any():
            new_out, new_mask = self.forward(new_out, start_times_tensor,
                                             time_periods_tensor, para_tensor,
                                             new_mask)
        return new_out

###############################################################################
# Quick unit‑test
###############################################################################
if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, D = 2, 8, 4
    out  = torch.randn(B, S, D, requires_grad=True)
    para = torch.randn(B, 2)
    start = torch.linspace(0, 3600 * 24, S).repeat(B, 1)
    dur   = torch.full_like(start, 600.0)
    mask  = torch.zeros
