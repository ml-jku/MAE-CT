import einops
import torch
import torch.nn as nn
from flash_attn.flash_attention import FlashAttention


class FlashAttentionWrapper(nn.Module):
    """ timm.vision_transformer.Attention but with FlashAttention """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # restriction from flash_attn.flash_attention.FlashMHA (could change in the future)
        assert head_dim % 8 == 0 and head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn = FlashAttention(softmax_scale=head_dim ** -0.5, attention_dropout=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # attention dropout for non FlashAttention forward
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)

        if qkv.dtype in [torch.float16, torch.bfloat16]:
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            x = self.attn(qkv)[0]
            x = einops.rearrange(x, "bs l n_heads head_dim -> bs l (n_heads head_dim)")
        else:
            # not all operations are mixed precision (e.g. automatic shape inferences)
            # copy pasted from timm.vision_transformer.Attention
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * self.attn.softmax_scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
