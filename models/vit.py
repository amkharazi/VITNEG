import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
# Patch embedding
# ---------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, bias: bool = True):
        super().__init__()
        assert isinstance(embed_dim, int), f"embed_dim must be int, got {type(embed_dim).__name__}"
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0, f"Input H,W must be divisible by patch_size={ps}"
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', h=ps, w=ps)
        x = self.projection(x)
        return x


# ---------------------------
# Multi-Head Self-Attention
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 bias: bool = True,
                 out_embed: bool = True,
                 softmax_type: str = "vanilla"):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_embed = out_embed
        if out_embed:
            self.fc_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._softmax = get_softmax_callable(softmax_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        Q = rearrange(self.query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(self.key(x),   'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(self.value(x), 'b n (h d) -> b h n d', h=self.num_heads)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = self._softmax(attn_logits, dim=-1)
        x = torch.matmul(attn, V)

        x = rearrange(x, 'b h n d -> b n (h d)')
        if self.out_embed:
            x = self.fc_out(x)
        return x


# ---------------------------
# DropPath (stochastic depth)
# ---------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ---------------------------
# Encoder block (Pre-LN)
# ---------------------------
class Encoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout: float = 0.1,
                 bias: bool = True,
                 out_embed: bool = True,
                 drop_path: float = 0.0,
                 softmax_type: str = "vanilla"):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            out_embed=out_embed,
            softmax_type=softmax_type,
        )
        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------
# Vision Transformer (ViT)
# ---------------------------
class VisionTransformer(nn.Module):
    def __init__(self,
                 input_size=(3, 224, 224),
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_dim: int = 3072,
                 dropout: float = 0.1,
                 bias: bool = True,
                 out_embed: bool = True,
                 drop_path: float = 0.1,
                 softmax_type: str = "vanilla"):
        super().__init__()

        C, H, W = input_size
        assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"
        num_patches = (H // patch_size) * (W // patch_size)

        self.patch_embedding = PatchEmbedding(
            in_channels=C,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=bias
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Encoder(embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    bias=bias,
                    out_embed=out_embed,
                    drop_path=drop_path,
                    softmax_type=softmax_type)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = x + self.pos_embedding

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls)


# ---------------------------
# Factory
# ---------------------------
def vit(**kwargs) -> VisionTransformer:
    """
    Vision Transformer factory with ViT-B/16 style defaults.
    Override any argument via kwargs.
    Example:
        model = vit(num_classes=10, softmax_type="neg")
    """
    defaults = dict(
        input_size=(3, 224, 224),
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        dropout=0.1,
        bias=True,
        out_embed=True,
        drop_path=0.1,
        softmax_type="vanilla",
    )
    defaults.update(kwargs)
    return VisionTransformer(**defaults)



# ---------------------------
# Quick sanity check + basic backprop
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_once(softmax_type: str):
        print(f"\n=== Sanity/backprop with softmax_type='{softmax_type}' ===")
        B, C, H, W = 2, 3, 224, 224
        num_classes = 10

        model = vit(input_size=(C, H, W),
                    patch_size=16,
                    num_classes=num_classes,
                    embed_dim=384,     # smaller for quick test
                    num_heads=6,
                    num_layers=2,
                    mlp_dim=768,
                    softmax_type=softmax_type).to(device)
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

        # (optional) check a grad norm isn’t NaN/zero
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
