import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------
# Softmax selector (supports "vanilla" and your sign-preserving variant "neg")
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
# DropPath (stochastic depth)
# ---------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


# ---------------------------
# Patch Embedding (conv stride=patch_size)
# ---------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x)                    # B,C,H',W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)    # B, HW, C
        x = self.norm(x)
        return x, (H, W)


# ---------------------------
# Patch Merging (downsample by 2, doubles channel dim)
# ---------------------------
class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for PatchMerging"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4C
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                    # B, (H/2*W/2), 2C
        return x


# ---------------------------
# Window helpers
# ---------------------------
def window_partition(x, window_size: int):
    # x: B, H, W, C
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  # (B * num_windows, ws, ws, C)

def window_reverse(windows, window_size: int, H: int, W: int):
    # windows: (B * num_windows, ws, ws, C)
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------
# Window Multi-Head Self-Attention with relative position bias
# ---------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size: Tuple[int, int], num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0,
                 softmax_type: str = "vanilla"):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias
        Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )  # (2Wh-1)*(2Ww-1), nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self._softmax = get_softmax_callable(softmax_type)

    def forward(self, x, mask=None):
        # x: (B*nW, N, C) where N=window_size*window_size
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, BnW, nH, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (BnW, nH, N, N)

        # relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # N, N, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # 1, nH, N, N
        attn = attn + relative_position_bias

        if mask is not None:
            # mask: (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(BnW // nW, nW, self.num_heads, N, N)
            # align dtype/device and broadcast over batch & heads: (1, nW, 1, N, N)
            mask = mask.to(dtype=attn.dtype, device=attn.device)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self._softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BnW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------
# Swin Transformer Block (W-MSA or SW-MSA)
# ---------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, softmax_type="vanilla"):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if the window size is larger than input, don't shift or window
            self.window_size = min(self.input_resolution)
            self.shift_size = 0
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, softmax_type=softmax_type
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        # attention mask for SW-MSA
        self.register_buffer("attn_mask", self._create_attn_mask())

    def _create_attn_mask(self):
        H, W = self.input_resolution
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        cnt = 0
        ws = self.window_size
        ss = self.shift_size
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)  # nW, N
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, N, N
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        ws = self.window_size
        ss = self.shift_size

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if ss > 0:
            shifted_x = torch.roll(x, shifts=(-ss, -ss), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, ws)          # (B*nW, ws, ws, C)
        x_windows = x_windows.view(-1, ws * ws, C)           # (B*nW, N, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, ws, ws, C)
        shifted_x = window_reverse(attn_windows, ws, H, W)   # B,H,W,C

        # reverse cyclic shift
        if ss > 0:
            x = torch.roll(shifted_x, shifts=(ss, ss), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------
# Swin Stage (multiple blocks + optional downsample)
# ---------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, downsample=True, softmax_type="vanilla"):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # drop path schedule across blocks (linear or per-list)
        if isinstance(drop_path, float):
            dpr = [drop_path] * depth
        else:
            dpr = drop_path

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    softmax_type=softmax_type
                )
            )

        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------
# Swin Transformer (defaults = Tiny config)
# ---------------------------
class SwinTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,               # base dim
                 depths=(2, 2, 6, 2),        # Tiny: (2,2,6,2)
                 num_heads=(3, 6, 12, 24),   # Tiny
                 window_size=7,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 softmax_type="vanilla"):
        super().__init__()

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # compute resolutions per stage
        H = W = img_size // patch_size
        self.layers = nn.ModuleList()

        # stochastic depth schedule across all blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        dim = embed_dim
        res = (H, W)
        for i_layer in range(len(depths)):
            depth = depths[i_layer]
            heads = num_heads[i_layer]
            downsample = (i_layer < len(depths) - 1)

            layer = BasicLayer(
                dim=dim,
                input_resolution=res,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dp_rates[cur:cur + depth],
                norm_layer=norm_layer,
                downsample=downsample,
                softmax_type=softmax_type
            )
            self.layers.append(layer)
            cur += depth

            if downsample:
                dim *= 2
                res = (res[0] // 2, res[1] // 2)

        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: B,C,H,W (H,W should be divisible by patch_size, and H',W' divisible by window sizes after shifts)
        x, (H, W) = self.patch_embed(x)  # B, HW, C
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # global average over tokens
        x = self.head(x)
        return x


# ---------------------------
# Factory
# ---------------------------
def swin(**kwargs) -> SwinTransformer:
    """
    Defaults to Swin-Tiny layout (embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=7).
    Tweak args to get Small/Base/Large.
    """
    defaults = dict(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        softmax_type="vanilla",
    )
    defaults.update(kwargs)
    return SwinTransformer(**defaults)


# ---------------------------
# Quick sanity check + basic backprop (ViT-style helper)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_once(softmax_type: str):
        print(f"\n=== Sanity/backprop with softmax_type='{softmax_type}' ===")
        B, C, H, W = 2, 3, 224, 224
        num_classes = 10

        model = swin(
            img_size=H,
            patch_size=4,
            in_chans=C,
            num_classes=num_classes,
            # Tiny-ish defaults kept; tweak if you want faster run
            embed_dim=96,
            depths=(2, 2, 2, 2),       # shallower for quicker sanity (you can switch back to (2,2,6,2))
            num_heads=(3, 6, 12, 24),
            window_size=7,
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
