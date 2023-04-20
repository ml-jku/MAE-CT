import logging
import os

import torch.nn as nn
from timm.models.vision_transformer import LayerScale, DropPath, Mlp, Attention

from utils.log_once import log_once


class VitBlock(nn.Module):
    """ timm.models.vision_transformer.Block that uses FlashAttention when possible """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm1_layer=nn.LayerNorm,
            norm2_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm1_layer(dim)
        attn_kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        if os.environ.get("DISABLE_FLASH_ATTENTION", "false").lower() in ("true", "1", "t"):
            self.attn = Attention(**attn_kwargs)
            log_once(lambda: logging.info(f"disabled FlashAttention via environment variable"), key="FlashAttention")
        else:
            try:
                from models.modules.flash_attention_wrapper import FlashAttentionWrapper
                self.attn = FlashAttentionWrapper(**attn_kwargs)
                log_once(lambda: logging.info(f"using FlashAttention"), key="FlashAttention")
            except ImportError:
                self.attn = Attention(**attn_kwargs)
                # log warning on linux
                if os.name != "nt":
                    log_once(lambda: logging.warning(f"no FlashAttention available"), key="FlashAttention")

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm2_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
