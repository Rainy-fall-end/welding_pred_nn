import torch
import torch.nn as nn
import math

# -----------------------------------------------------------------------------
# Utility Modules
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = None, drop: float = 0.):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 * (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
# Patch Embed / Reconstruction
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Image → patch sequence"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)          # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class PatchReconstruct(nn.Module):
    """patch sequence → image"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768, out_chans: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, D = x.shape
        h = w = int(math.sqrt(N))
        x = x.transpose(1, 2).reshape(B, D, h, w)
        x = self.proj(x)
        return x  # (B, out_chans, H, W)


# -----------------------------------------------------------------------------
# ViT Encoder / Decoder (single frame)
# -----------------------------------------------------------------------------
class ViTEncoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
                 depth: int = 8, num_heads: int = 8, mlp_ratio: float = 4., drop: float = 0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # (B,C,H,W)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x  # (B, N, D)


class ViTDecoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768,
                 depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4., drop: float = 0., out_chans: int = 3):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.reconstruct = PatchReconstruct(img_size, patch_size, embed_dim, out_chans)

    def forward(self, x):  # (B,N,D)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.reconstruct(x)
        return x  # (B,C,H,W)


# -----------------------------------------------------------------------------
# Frame-level Autoencoder (wrapper of encoder / decoder)
# -----------------------------------------------------------------------------
class ViTAutoencoder(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
                 enc_depth: int = 8, dec_depth: int = 4, enc_heads: int = 8, dec_heads: int = 8, mlp_ratio: float = 4.):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_chans, embed_dim, enc_depth, enc_heads, mlp_ratio)
        self.decoder = ViTDecoder(img_size, patch_size, embed_dim, dec_depth, dec_heads, mlp_ratio, out_chans=in_chans)

    def forward(self, x):  # (B,C,H,W)
        latent = self.encoder(x)          # (B, N, D)
        recon = self.decoder(latent)      # (B, C, H, W)
        return recon, latent


# -----------------------------------------------------------------------------
# Sequence-level Autoencoder
# -----------------------------------------------------------------------------
class SequenceViTAutoencoder(nn.Module):
    """Handle input of shape (B, T, C, H, W) and output (B, T, D)"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768,
                 enc_depth: int = 8, dec_depth: int = 4, enc_heads: int = 8, dec_heads: int = 8, mlp_ratio: float = 4.):
        super().__init__()
        # Frame-wise ViTAE shared across time
        self.frame_ae = ViTAutoencoder(img_size, patch_size, in_chans, embed_dim,
                                       enc_depth, dec_depth, enc_heads, dec_heads, mlp_ratio)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        # Map frame-level embedding → patch sequence for reconstruction
        self.expand = nn.Linear(embed_dim, embed_dim * self.n_patches)
        # Optional temporal modeling head (identity by default)
        self.temporal_head = nn.Identity()

    def encode(self, x):  # (B,T,C,H,W) → (B,T,D)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        latent_patches = self.frame_ae.encoder(x_flat)        # (B*T, N, D)
        latent_frames = latent_patches.mean(dim=1)            # (B*T, D)
        latent_frames = latent_frames.view(B, T, self.embed_dim)
        latent_frames = self.temporal_head(latent_frames)     # (B,T,D)
        return latent_frames

    def decode(self, z):   # (B,T,D) → (B,T,C,H,W)
        B, T, D = z.shape
        z_flat = z.view(B * T, D)                             # (B*T, D)
        patches = self.expand(z_flat).view(B * T, self.n_patches, D)  # (B*T, N, D)
        recon_flat = self.frame_ae.decoder(patches)           # (B*T, C, H, W)
        C = recon_flat.size(1)
        recon = recon_flat.view(B, T, C, self.img_size, self.img_size)
        return recon

    def forward(self, x):  # x: (B,T,C,H,W)
        latent_seq = self.encode(x)           # (B,T,D)
        recon_seq = self.decode(latent_seq)   # (B,T,C,H,W)
        return recon_seq, latent_seq


# -----------------------------------------------------------------------------
# Quick sanity check
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, C, H, W = 2, 5, 3, 128, 128
    video = torch.randn(B, T, C, H, W)
    model = SequenceViTAutoencoder(img_size=128, patch_size=16, in_chans=3, embed_dim=256,
                                   enc_depth=6, dec_depth=3, enc_heads=8, dec_heads=8)
    recon, latent = model(video)
    print("recon:", recon.shape, "latent:", latent.shape)  # (2,5,3,128,128) (2,5,256)
