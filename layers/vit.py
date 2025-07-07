from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Helper: simple MLP used inside Transformer blocks
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

# -----------------------------------------------------------------------------
# Attention + Transformer Block
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# -----------------------------------------------------------------------------
# Patch Embedding that stores padding info
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Converts HxW images to (N, D) patch embeddings, keeps padding meta"""

    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size  # (ph, pw)
        ph, pw = patch_size

        # maximum grid size based on raw img_size (for pos_embed shape)
        gh = (img_size[0] + ph - 1) // ph  # ceil division
        gw = (img_size[1] + pw - 1) // pw
        self.n_patches = gh * gw

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """Return patch embeddings and padding meta
        x : (B, C, H, W)
        returns patches : (B, N, D)  where N == gh*gw after padding
                pad_info: (H_orig, W_orig, pad_h, pad_w)
        """
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw

        x = F.pad(x, (0, pad_w, 0, pad_h))  # (L, R, T, B)
        x = self.proj(x)                    # (B, D, H', W')
        patches = x.flatten(2).transpose(1, 2)  # (B, N, D)

        pad_info = (H, W, pad_h, pad_w)
        return patches, pad_info

# -----------------------------------------------------------------------------
# Reconstruct image from patch embeddings, remove padding
# -----------------------------------------------------------------------------
class PatchReconstruct(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], embed_dim: int, out_chans: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, patches: torch.Tensor, pad_info: Tuple[int, int, int, int]):
        """patches : (B, N, D)"""
        H_orig, W_orig, pad_h, pad_w = pad_info
        ph, pw = self.patch_size
        B, N, D = patches.shape
        gh = (H_orig + pad_h) // ph
        gw = (W_orig + pad_w) // pw
        assert N == gh * gw, "patch count mismatch"

        x = patches.transpose(1, 2).reshape(B, D, gh, gw)  # (B, D, gh, gw)
        x = self.proj(x)                                   # (B, C, H_pad, W_pad)
        return x[:, :, :H_orig, :W_orig]

# -----------------------------------------------------------------------------
# ViT Encoder / Decoder (single frame)
# -----------------------------------------------------------------------------
class ViTEncoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], in_chans: int, embed_dim: int,
                 depth: int, num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        patches, pad_info = self.patch_embed(x)          # (B, N, D)
        patches = patches + self.pos_embed[:, :patches.size(1)]
        patches = self.blocks(patches)
        return self.norm(patches), pad_info              # (B, N, D), pad_info

class ViTDecoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], embed_dim: int, out_chans: int,
                 depth: int, num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        # pos_embed length must match the maximum possible patches
        ph, pw = patch_size
        gh = (img_size[0] + ph - 1) // ph
        gw = (img_size[1] + pw - 1) // pw
        n_patches = gh * gw
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruct = PatchReconstruct(patch_size, embed_dim, out_chans)

    def forward(self, patches: torch.Tensor, pad_info):
        patches = patches + self.pos_embed[:, :patches.size(1)]
        patches = self.blocks(patches)
        patches = self.norm(patches)
        return self.reconstruct(patches, pad_info)

# -----------------------------------------------------------------------------
# Frame-level Autoencoder
# -----------------------------------------------------------------------------
class ViTAutoencoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], in_chans: int, embed_dim: int,
                 enc_depth: int, dec_depth: int, enc_heads: int, dec_heads: int, mlp_ratio: float):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, enc_depth, enc_heads, mlp_ratio, 0.)
        self.decoder = ViTDecoder(img_size, patch_size, embed_dim, in_chans, dec_depth, dec_heads, mlp_ratio, 0.)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)  # (patch_latent, pad_info)

    def decode(self, patch_latent: torch.Tensor, pad_info):
        return self.decoder(patch_latent, pad_info)

    def forward(self, x: torch.Tensor):
        patch_latent, pad_info = self.encode(x)
        recon = self.decode(patch_latent, pad_info)
        return recon, patch_latent

# -----------------------------------------------------------------------------
# Sequence-level Autoencoder (B, T, C, H, W) or (B, C, H, W)
# -----------------------------------------------------------------------------
class SequenceViTAutoencoder(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int] = (224, 224),
                 patch_size: Tuple[int, int] = (16, 16),
                 in_chans: int = 3,
                 embed_dim: int = 256,
                 enc_depth: int = 8,
                 dec_depth: int = 4,
                 enc_heads: int = 8,
                 dec_heads: int = 8,
                 mlp_ratio: float = 4.):
        super().__init__()
        self.frame_ae = ViTAutoencoder(img_size, patch_size, in_chans, embed_dim,
                                       enc_depth, dec_depth, enc_heads, dec_heads, mlp_ratio)
        ph, pw = patch_size
        gh = (img_size[0] + ph - 1) // ph
        gw = (img_size[1] + pw - 1) // pw
        self.n_patches = gh * gw
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.patch_expand = nn.Linear(embed_dim, embed_dim * self.n_patches)
        self.temporal_head = nn.Identity()

    # ---------- encode ----------
    def encode(self, x: torch.Tensor):
        """Return latent (B, T, D) or (B, D) and pad_info"""
        if x.dim() == 5:  # video
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            patch_latent, pad_info = self.frame_ae.encode(x_flat)  # (B*T, N, D)
            frame_latent = patch_latent.mean(1).reshape(B, T, self.embed_dim)
            return self.temporal_head(frame_latent), pad_info
        elif x.dim() == 4:  # single frame
            patch_latent, pad_info = self.frame_ae.encode(x)       # (B, N, D)
            frame_latent = patch_latent.mean(1)
            return self.temporal_head(frame_latent), pad_info
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

    # ---------- decode ----------
    def decode(self, latent: torch.Tensor, pad_info):
        if latent.dim() == 3:  # (B, T, D)
            B, T, D = latent.shape
            patches = self.patch_expand(latent.reshape(B * T, D)).view(B * T, self.n_patches, D)
            recon = self.frame_ae.decode(patches, pad_info)        # (B*T, C, H, W)
            C = recon.size(1)
            return recon.view(B, T, C, self.img_size[0], self.img_size[1])
        elif latent.dim() == 2:  # (B, D)
            B, D = latent.shape
            patches = self.patch_expand(latent).view(B, self.n_patches, D)
            recon = self.frame_ae.decode(patches, pad_info)        # (B, C, H, W)
            return recon
        else:
            raise ValueError(f"Unsupported latent shape {latent.shape}")

    # ---------- forward ----------
    def forward(self, x: torch.Tensor):
        latent, pad_info = self.encode(x)
        recon = self.decode(latent, pad_info)
        return recon, latent
