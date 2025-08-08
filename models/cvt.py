import math
import warnings
from typing import Tuple, Sequence

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
# Overlapping Conv Patch Embedding (per CvT stage)
# ---------------------------
class ConvEmbed(nn.Module):
    """
    Overlapping conv embedding:
      Conv2d(in_ch, embed_dim, kernel_size=k, stride=s, padding=k//2) -> flatten -> LayerNorm
    """
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int, norm_layer=nn.LayerNorm):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                   # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H'*W', D)
        x = self.norm(x)
        return x, (H, W)


# ---------------------------
# Depthwise Conv Projection used inside attention (on Q/K/V)
# ---------------------------
class DWConv2dProj(nn.Module):
    """
    Depthwise convolution on token maps (B, N, C) by reshaping back to (B,C,H,W).
    Allows stride > 1 for spatial reduction (used for K/V).
    """
    def __init__(self, dim: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.stride = stride
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride,
                                padding=pad, groups=dim, bias=True)

    def forward(self, x, H: int, W: int):
        # x: (B, N, C) with N = H*W
        B, N, C = x.shape
        assert N == H * W, "Token count must match H*W"
        x = x.transpose(1, 2).view(B, C, H, W)    # (B, C, H, W)
        x = self.dwconv(x)                         # (B, C, Hs, Ws)
        B, C, Hs, Ws = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, Hs*Ws, C)
        return x, (Hs, Ws)


# ---------------------------
# Convolutional Multi-Head Self-Attention (CvT-style)
# ---------------------------
class ConvMHSA(nn.Module):
    """
    CvT attention:
      - Linear q/k/v
      - Depthwise conv applied to q/k/v (k/v often downsampled by sr_ratio)
      - MHSA with optional reduced K/V sequence length
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 q_kernel: int = 3, kv_kernel: int = 3,
                 q_stride: int = 1, kv_stride: int = 1,
                 softmax_type: str = "vanilla"):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # depthwise conv projections
        self.q_proj = DWConv2dProj(dim, kernel_size=q_kernel, stride=q_stride)
        self.k_proj = DWConv2dProj(dim, kernel_size=kv_kernel, stride=kv_stride)
        self.v_proj = DWConv2dProj(dim, kernel_size=kv_kernel, stride=kv_stride)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._softmax = get_softmax_callable(softmax_type)

    def _reshape_heads(self, x):
        # (B, N, C) -> (B, heads, N, head_dim)
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def forward(self, x, H: int, W: int):
        # x: (B, N, C), N=H*W
        B, N, C = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # depthwise conv projections (q stride=1; k/v can downsample spatially)
        q, (Hq, Wq) = self.q_proj(q, H, W)               # (B, Nq, C)
        k, (Hk, Wk) = self.k_proj(k, H, W)               # (B, Nk, C)
        v, _         = self.v_proj(v, H, W)              # (B, Nk, C)

        q = self._reshape_heads(q)                       # (B, h, Nq, d)
        k = self._reshape_heads(k)                       # (B, h, Nk, d)
        v = self._reshape_heads(v)                       # (B, h, Nk, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, h, Nq, Nk)
        attn = self._softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                   # (B, h, Nq, d)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, C)  # (B, Nq, C)

        # If q_stride==1, Nq == H*W; else it's reduced; here we assume q_stride==1 (typical in CvT)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, (Hq, Wq)


# ---------------------------
# Transformer Block (Pre-LN) with ConvMHSA
# ---------------------------
class CvTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 q_kernel: int = 3, kv_kernel: int = 3, q_stride: int = 1, kv_stride: int = 1,
                 softmax_type: str = "vanilla"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ConvMHSA(dim, num_heads, qkv_bias, attn_drop, drop,
                             q_kernel=q_kernel, kv_kernel=kv_kernel,
                             q_stride=q_stride, kv_stride=kv_stride,
                             softmax_type=softmax_type)
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
        # Attention
        x_attn_in = self.norm1(x)
        attn_out, (Hn, Wn) = self.attn(x_attn_in, H, W)
        x = x + self.drop_path(attn_out)

        # If attention changed resolution (via q_stride), update H,W; we use q_stride=1 so keep them
        H, W = Hn, Wn

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, (H, W)


# ---------------------------
# CvT Stage (ConvEmbed + multiple CvTBlocks)
# ---------------------------
class CvTStage(nn.Module):
    def __init__(self,
                 in_chans: int,
                 embed_dim: int,
                 depth: int,
                 num_heads: int,
                 patch_kernel: int,
                 patch_stride: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 drop_path: Sequence[float] = (0.0,),
                 q_kernel: int = 3,
                 kv_kernel: int = 3,
                 q_stride: int = 1,
                 kv_stride: int = 1,
                 norm_layer=nn.LayerNorm,
                 softmax_type: str = "vanilla"):
        super().__init__()

        self.embed = ConvEmbed(in_chans, embed_dim, kernel_size=patch_kernel, stride=patch_stride, norm_layer=norm_layer)

        # blocks
        dpr = list(drop_path)
        assert len(dpr) == depth
        self.blocks = nn.ModuleList([
            CvTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=dpr[i],
                     q_kernel=q_kernel, kv_kernel=kv_kernel, q_stride=q_stride, kv_stride=kv_stride,
                     norm_layer=norm_layer, softmax_type=softmax_type)
            for i in range(depth)
        ])

    def forward(self, x):
        # x: (B, C_in, H, W)
        x, (H, W) = self.embed(x)              # tokens for this stage
        for blk in self.blocks:
            x, (H, W) = blk(x, H, W)
        # return tokens as (B,N,C) and also a 4D map for next stage's conv stem
        B, N, C = x.shape
        fmap = x.transpose(1, 2).view(B, C, H, W)
        return x, (H, W), fmap


# ---------------------------
# CvT (3-stage default, Tiny-ish)
# ---------------------------
class CvT(nn.Module):
    """
    A compact CvT similar to CvT-13-ish layout (lightened for sanity tests by default):
      Stage 1: ConvEmbed(7x7, stride 4), dim 64,  depth 1,  heads 1,  kv_stride=1
      Stage 2: ConvEmbed(3x3, stride 2), dim 192, depth 2,  heads 3,  kv_stride=2 (reduce K/V)
      Stage 3: ConvEmbed(3x3, stride 2), dim 384, depth 6,  heads 6,  kv_stride=2
    You can scale depths/heads/dims via the factory.
    """
    def __init__(self,
                 img_size: int = 224,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dims: Sequence[int] = (64, 192, 384),
                 depths: Sequence[int] = (1, 2, 6),          # light default
                 num_heads: Sequence[int] = (1, 3, 6),
                 patch_kernels: Sequence[int] = (7, 3, 3),
                 patch_strides: Sequence[int] = (4, 2, 2),
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1,
                 q_kernels: Sequence[int] = (3, 3, 3),
                 kv_kernels: Sequence[int] = (3, 3, 3),
                 q_strides: Sequence[int] = (1, 1, 1),
                 kv_strides: Sequence[int] = (1, 2, 2),     # spatial reduction for K/V in later stages
                 norm_layer=nn.LayerNorm,
                 softmax_type: str = "vanilla"):
        super().__init__()

        assert len(embed_dims) == len(depths) == len(num_heads) == len(patch_kernels) == len(patch_strides)
        assert len(q_kernels) == len(kv_kernels) == len(q_strides) == len(kv_strides) == len(depths)

        self.num_stages = len(embed_dims)

        # stochastic depth distribution across ALL blocks
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        stages = []
        cur = 0
        in_channels = in_chans
        for i in range(self.num_stages):
            stages.append(
                CvTStage(
                    in_chans=in_channels,
                    embed_dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    patch_kernel=patch_kernels[i],
                    patch_stride=patch_strides[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur:cur + depths[i]],
                    q_kernel=q_kernels[i],
                    kv_kernel=kv_kernels[i],
                    q_stride=q_strides[i],
                    kv_stride=kv_strides[i],
                    norm_layer=norm_layer,
                    softmax_type=softmax_type,
                )
            )
            cur += depths[i]
            in_channels = embed_dims[i]  # next stage's input channels

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
        # x: (B, C, H, W)
        fmap = x
        tokens = None
        H = W = None
        for stage in self.stages:
            tokens, (H, W), fmap = stage(fmap)  # tokens: (B, N, C)
        x = self.norm(tokens)
        x = x.mean(dim=1)            # global average over tokens
        x = self.head(x)
        return x


# ---------------------------
# Factory
# ---------------------------
def cvt(**kwargs) -> CvT:
    """
    CvT factory. Defaults to a light 3-stage config (roughly CvT-Tiny-ish).
    Override via kwargs for deeper/bigger models:
      - For a closer CvT-13: embed_dims=(64,192,384), depths=(1,2,10), heads=(1,3,6)
      - For stronger reduction: kv_strides=(1,2,2) or (2,2,2)
    """
    defaults = dict(
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=(64, 192, 384),
        depths=(1, 2, 6),
        num_heads=(1, 3, 6),
        patch_kernels=(7, 3, 3),
        patch_strides=(4, 2, 2),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        q_kernels=(3, 3, 3),
        kv_kernels=(3, 3, 3),
        q_strides=(1, 1, 1),
        kv_strides=(1, 2, 2),      # downsample K/V in later stages
        softmax_type="vanilla",
    )
    defaults.update(kwargs)
    return CvT(**defaults)


# ---------------------------
# Quick sanity check + basic backprop (same style as your ViT helper)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_once(softmax_type: str):
        print(f"\n=== CvT sanity/backprop with softmax_type='{softmax_type}' ===")
        B, C, H, W = 2, 3, 224, 224
        num_classes = 10

        model = cvt(
            img_size=H,
            in_chans=C,
            num_classes=num_classes,
            # Light config for a quick pass; bump depths for heavier tests
            embed_dims=(64, 192, 384),
            depths=(1, 2, 4),
            num_heads=(1, 3, 6),
            kv_strides=(1, 2, 2),
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
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print("grad norm:", total_norm)

        # step
        optim.step()

        # quick second forward to show loss can change
        with torch.no_grad():
            loss2 = criterion(model(x), y)
        print("loss (post-step):", float(loss2.cpu()))

    run_once("vanilla")
    run_once("neg")
