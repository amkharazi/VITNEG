import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = rearrange(x, "b c h w -> b (h w) c")  # Shape: (B, num_patches, embed_dim)
        return x

# Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # Shape: (B, N, 3 * dim)
        qkv = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv

        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k) * self.scale  # Shape: (B, h, N, N)
        # print(f'hereX1 {attn.shape}')
        attn = attn.softmax(dim=-1)
        # print(f'hereX2 {attn[0][2].sum(dim = -1).shape}')
        # print(f'hereX3 {attn[0][2].sum(dim = -1)}')

        out = torch.einsum("bhqk, bhvd -> bhqd", attn, v)  # Shape: (B, h, N, d)
        out = rearrange(out, "b h n d -> b n (h d)")  # Shape: (B, N, C)
        return self.fc_out(out)

# MLP Layer
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32,
                  patch_size=4,
                    in_channels=3,
                      num_classes=10,
                        dim=64,
                          depth=6,
                            heads=8,
                              mlp_dim=128,
                                dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.positional_encoding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, dim))
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_encoding

        for block in self.transformer:
            x = block(x)

        cls_output = x[:, 0]
        return self.mlp_head(cls_output)