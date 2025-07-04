import torch
import torch.nn as nn
from typing import Tuple

# -----------------------------------------------------------------------------
# Utility Modules
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = None, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim or in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


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
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
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
# Patch Embedding / Reconstruction (supports non‑square)
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], in_chans: int, embed_dim: int):
        super().__init__()
        ih, iw = img_size; ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0, "Image dimensions must be divisible by patch size"
        self.grid_size = (ih // ph, iw // pw)
        self.n_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B,D,gh,gw)
        x = x.flatten(2).transpose(1, 2)      # (B,N,D)
        return x


class PatchReconstruct(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], embed_dim: int, out_chans: int):
        super().__init__()
        ih, iw = img_size; ph, pw = patch_size
        self.grid_size = (ih // ph, iw // pw)
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, D = x.shape
        gh, gw = self.grid_size
        assert N == gh * gw, "patch count mismatch"
        x = x.transpose(1, 2).reshape(B, D, gh, gw)
        return self.proj(x)


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

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)


class ViTDecoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], embed_dim: int, out_chans: int,
                 depth: int, num_heads: int, mlp_ratio: float, drop: float):
        super().__init__()
        gh, gw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruct = PatchReconstruct(img_size, patch_size, embed_dim, out_chans)

    def forward(self, x):
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.reconstruct(x)


# -----------------------------------------------------------------------------
# Frame-level Autoencoder
# -----------------------------------------------------------------------------
class ViTAutoencoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: Tuple[int, int], in_chans: int, embed_dim: int,
                 enc_depth: int, dec_depth: int, enc_heads: int, dec_heads: int, mlp_ratio: float):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, enc_depth, enc_heads, mlp_ratio, 0.)
        self.decoder = ViTDecoder(img_size, patch_size, embed_dim, in_chans, dec_depth, dec_heads, mlp_ratio, 0.)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


# -----------------------------------------------------------------------------
# Sequence-level Autoencoder (B,T,C,H,W)
# -----------------------------------------------------------------------------
class SequenceViTAutoencoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: Tuple[int, int] = (16, 16), in_chans: int = 3,
                 embed_dim: int = 256, enc_depth: int = 8, dec_depth: int = 4, enc_heads: int = 8, dec_heads: int = 8,
                 mlp_ratio: float = 4.):
        super().__init__()
        self.frame_ae = ViTAutoencoder(img_size, patch_size, in_chans, embed_dim,
                                       enc_depth, dec_depth, enc_heads, dec_heads, mlp_ratio)
        gh, gw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.n_patches = gh * gw
        self.embed_dim = embed_dim
        self.expand = nn.Linear(embed_dim, embed_dim * self.n_patches)
        self.temporal_head = nn.Identity()
        self.img_size = img_size

    def encode(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        latent_patch = self.frame_ae.encoder(x_flat)          # (B*T,N,D)
        latent_vec = latent_patch.mean(1)                     # (B*T,D)
        latent_vec = latent_vec.view(B, T, self.embed_dim)
        return self.temporal_head(latent_vec)

    def decode(self, z):
        B, T, D = z.shape
        patches = self.expand(z.reshape(B * T, D)).view(B * T, self.n_patches, D)
        recon = self.frame_ae.decoder(patches)               # (B*T,C,H,W)
        C = recon.size(1)
        return recon.view(B, T, C, self.img_size[0], self.img_size[1])

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


# -----------------------------------------------------------------------------
# Sanity check with non‑square frames
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, C, H, W = 2, 4, 3, 96, 128  # non‑square 96×128
    video = torch.randn(B, T, C, H, W)
    model = SequenceViTAutoencoder(img_size=(H, W), patch_size=(16, 16), in_chans=C, embed_dim=256,
                                   enc_depth=6, dec_depth=3)
    out, lat = model(video)
    print(out.shape, lat.shape)  # (2,4,3,96,128) (2,4,256)
