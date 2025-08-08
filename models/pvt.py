import math
import warnings
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Softmax selector (supports "vanilla" and sign-preserving "neg")
# ---------------------------
def get_softmax_callable(softmax_type: str):
    if softmax_type == "vanilla":
        return lambda x, dim=-1: F.softmax(x, dim=dim)
    elif softmax_type == "neg":  # sign-preserve: sign(attn) * softmax(|attn|)
        return lambda x, dim=-1: torch.sign(x) * F.softmax(torch.abs(x), dim=dim)
    else:
        warnings.warn(f"Unknown softmax_type='{softmax_type}'. Falling back to vanilla softmax.", RuntimeWarning)
        return lambda x, dim=-1: F.softmax(x, dim=dim)


# ---------------------------
# Stochastic depth (DropPath)
# ---------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


# ---------------------------
# Overlapping Conv Patch Embedding (PVT uses strided conv stems)
# ---------------------------
class PatchEmbed(nn.Module):
    """
    Conv2d(in_ch, embed_dim, k, s, p) -> flatten -> LayerNorm
    Stage 1 often uses (7,4,3); later stages (3,2,1).
    """
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int, padding: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x)                    # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, H'*W', D)
        x = self.norm(x)
        return x, (H, W)


# ---------------------------
# Spatial Reduction Attention (SRA) as in PVT
# ---------------------------
class SRMHA(nn.Module):
    """
    Multi-head attention with spatial reduction on K/V via strided conv (sr_ratio).
    If sr_ratio > 1, we apply a Conv2d(stride=sr_ratio) on tokens reshaped to (B,C,H,W),
    then LayerNorm, then make k and v from the reduced sequence.
    """
    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, softmax_type: str = "vanilla"):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding=0, groups=dim, bias=True)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._softmax = get_softmax_callable(softmax_type)

    def _reshape_heads(self, x):
        # (B, N, C) -> (B, heads, N, head_dim)
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x, H: int, W: int):
        # x: (B, N, C), N = H * W
        B, N, C = x.shape
        q = self._reshape_heads(self.q(x))              # (B, h, N, d)

        if self.sr is not None:
            # reduce spatial tokens for K/V
            x_ = x.transpose(1, 2).view(B, C, H, W)    # (B, C, H, W)
            x_ = self.sr(x_)                            # (B, C, H', W'), depthwise conv
            Hk, Wk = x_.shape[2], x_.shape[3]
            x_ = x_.flatten(2).transpose(1, 2)          # (B, N', C)
            x_ = self.norm(x_)                          # LN on reduced tokens
            k = self._reshape_heads(self.k(x_))         # (B, h, N', d)
            v = self._reshape_heads(self.v(x_))         # (B, h, N', d)
        else:
            k = self._reshape_heads(self.k(x))
            v = self._reshape_heads(self.v(x))

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, N, N' or N)
        attn = self._softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                  # (B, h, N, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------
# Transformer Block (Pre-LN) for PVT
# ---------------------------
class PVTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 8.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, sr_ratio: int = 1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, softmax_type: str = "vanilla"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SRMHA(dim, num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias,
                          attn_drop=attn_drop, proj_drop=drop, softmax_type=softmax_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------
# PVT Stage (PatchEmbed + multiple PVTBlocks)
# ---------------------------
class PVTStage(nn.Module):
    def __init__(self,
                 in_chans: int,
                 embed_dim: int,
                 depth: int,
                 num_heads: int,
                 patch_kernel: int,
                 patch_stride: int,
                 patch_pad: int,
                 mlp_ratio: float,
                 sr_ratio: int,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: Sequence[float] = (0.0,),
                 norm_layer=nn.LayerNorm,
                 softmax_type: str = "vanilla"):
        super().__init__()

        self.embed = PatchEmbed(in_chans, embed_dim, kernel_size=patch_kernel, stride=patch_stride, padding=patch_pad, norm_layer=norm_layer)

        dpr = list(drop_path)
        assert len(dpr) == depth
        self.blocks = nn.ModuleList([
            PVTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=dpr[i],
                     sr_ratio=sr_ratio, norm_layer=norm_layer, softmax_type=softmax_type)
            for i in range(depth)
        ])

    def forward(self, x):
        # x: (B, C_in, H, W)
        x, (H, W) = self.embed(x)          # (B, N, C), (H, W)
        for blk in self.blocks:
            x = blk(x, H, W)
        fmap = x.transpose(1, 2).view(x.shape[0], x.shape[-1], H, W)  # for next stage's stem
        return x, (H, W), fmap


# ---------------------------
# Pyramid Vision Transformer (PVT)
# ---------------------------
class PVT(nn.Module):
    """
    Default config ~ PVT-Tiny style:
      - embed_dims = (64, 128, 320, 512)
      - depths     = (2, 2, 2, 2)
      - num_heads  = (1, 2, 5, 8)
      - sr_ratios  = (8, 4, 2, 1)  # heavier reduction earlier
      - mlp_ratio  = 8.0 (PVT often uses larger MLP)
    """
    def __init__(self,
                 img_size: int = 224,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dims: Sequence[int] = (64, 128, 320, 512),
                 depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (1, 2, 5, 8),
                 sr_ratios: Sequence[int] = (8, 4, 2, 1),
                 mlp_ratio: float = 8.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 softmax_type: str = "vanilla"):
        super().__init__()

        assert len(embed_dims) == len(depths) == len(num_heads) == len(sr_ratios) == 4

        # Stage stems (kernels/strides per PVT)
        patch_kernels = (7, 3, 3, 3)
        patch_strides = (4, 2, 2, 2)
        patch_pads    = (3, 1, 1, 1)

        # Distribute DropPath over all blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0

        in_channels = in_chans
        stages = []
        for i in range(4):
            stages.append(
                PVTStage(
                    in_chans=in_channels,
                    embed_dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    patch_kernel=patch_kernels[i],
                    patch_stride=patch_strides[i],
                    patch_pad=patch_pads[i],
                    mlp_ratio=mlp_ratio,
                    sr_ratio=sr_ratios[i],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur:cur + depths[i]],
                    norm_layer=norm_layer,
                    softmax_type=softmax_type
                )
            )
            cur += depths[i]
            in_channels = embed_dims[i]

        self.stages = nn.ModuleList(stages)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        fmap = x
        tokens = None
        for stage in self.stages:
            tokens, (H, W), fmap = stage(fmap)
        x = self.norm(tokens)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# ---------------------------
# Factory
# ---------------------------
def pvt(**kwargs) -> PVT:
    """
    PVT factory (defaults ~ Tiny).
    Change embed_dims/depths/heads/sr_ratios for Small/Medium/Large variants.
    """
    defaults = dict(
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=(64, 128, 320, 512),
        depths=(2, 2, 2, 2),
        num_heads=(1, 2, 5, 8),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratio=8.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        softmax_type="vanilla",
    )
    defaults.update(kwargs)
    return PVT(**defaults)


# ---------------------------
# Quick sanity check + basic backprop (ViT-style helper)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_once(softmax_type: str):
        print(f"\n=== PVT sanity/backprop with softmax_type='{softmax_type}' ===")
        B, C, H, W = 2, 3, 224, 224
        num_classes = 10

        model = pvt(
            img_size=H,
            in_chans=C,
            num_classes=num_classes,
            # keep defaults (Tiny-ish); tweak for speed:
            depths=(2, 2, 2, 2),           # can drop to (1,1,1,1) for super quick test
            softmax_type=softmax_type
        ).to(device)
        model.train()

        x = torch.randn(B, C, H, W, device=device)
        y = torch.randint(0, num_classes, (B,), device=device)

        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

        # forward
        logits = model(x)
        loss = criterion(logits, y)
        print("loss (pre-step):", float(loss.detach().cpu()))

        # backward
        optim.zero_grad(set_to_none=True)
        loss.backward()

        # grad norm check
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.data.norm(2).item()
                total_norm += g * g
        total_norm = math.sqrt(total_norm)
        print("grad norm:", total_norm)

        # step
        optim.step()

        # quick second forward
        with torch.no_grad():
            loss2 = criterion(model(x), y)
        print("loss (post-step):", float(loss2.cpu()))

    run_once("vanilla")
    run_once("neg")
