from functools import partial

import einops
import torch
import torch.nn as nn

from initializers.functional import initialize_layernorm_as_noaffine, initialize_linear_bias_to_zero
from models.base.single_model_base import SingleModelBase
from models.modules.batchnorm1d_vit import BatchNorm1dViT
from models.modules.nonshared_patch_embed import NonsharedPatchEmbed
from models.modules.patch_embed import PatchEmbed
from models.modules.vit_block import VitBlock
from models.modules.vit_conv_stem import ViTConvStem
from models.poolings.single_pooling import SinglePooling
from utils.formatting_util import list_to_string
from utils.param_checking import to_2tuple
from utils.positional_embedding import (
    get_2d_sincos_pos_embed,
    interpolate_pos_embed_permanent,
    interpolate_pos_embed_temporary,
)
from utils.vit_util import get_sequence_lengths


class VitMae(SingleModelBase):
    def __init__(
            self,
            patch_size,
            embedding_dim,
            depth,
            attention_heads,
            drop_path_rate=0.,
            drop_path_decay=False,
            patch_embed="shared",
            dropout=0.,
            cls_token=True,
            use_bn_in_mlp=False,
            use_bn_in_attn=False,
            use_learnable_pos_embed=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = to_2tuple(patch_size)
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.attention_heads = attention_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.use_learnable_pos_embed = use_learnable_pos_embed

        # initialize patch_embed
        c, h, w = self.input_shape
        patch_embed_kwargs = dict(img_size=(h, w), patch_size=self.patch_size, in_chans=c, embed_dim=embedding_dim)
        if patch_embed == "shared":
            self.patch_embed = PatchEmbed(**patch_embed_kwargs)
        elif patch_embed == "nonshared":
            self.patch_embed = NonsharedPatchEmbed(**patch_embed_kwargs)
        elif patch_embed == "conv":
            self.patch_embed = ViTConvStem(**patch_embed_kwargs)
            self.depth -= 1
        else:
            raise NotImplementedError
        self.pos_drop = nn.Dropout(p=dropout)

        # many MAE implementations don't use a CLS token
        self.n_aux_tokens = 0
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.n_aux_tokens += 1
        else:
            self.cls_token = None
        self.output_shape = (self.patch_embed.num_patches + self.n_aux_tokens, embedding_dim)

        if use_learnable_pos_embed:
            # learnable pos_embed
            # (timm ViT uses a position for cls token but this doesnt make sense as the cls token is learnable anyways)
            # this way interpolation of pos_embed is cleaner as the cls token pos_embed doesn't need to be removed
            # before the interpolation
            self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embedding_dim))
            self.logger.info(f"using learnable positional embedding")
        else:
            # fixed pos_embed
            self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches, embedding_dim))
            self.logger.info(f"using fixed positional embedding")

        norm1_layer = partial(BatchNorm1dViT if use_bn_in_attn else nn.LayerNorm, eps=1e-6)
        norm2_layer = partial(BatchNorm1dViT if use_bn_in_mlp else nn.LayerNorm, eps=1e-6)
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
            self.logger.info(f"using drop_path_decay: {list_to_string(dpr)}")
        else:
            dpr = [drop_path_rate] * self.depth
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=embedding_dim,
                num_heads=attention_heads,
                qkv_bias=True,
                norm1_layer=norm1_layer,
                norm2_layer=norm2_layer,
                drop_path=dpr[i],
                drop=dropout,
            )
            for i in range(self.depth)
        ])
        last_norm = partial(BatchNorm1dViT if use_bn_in_attn or use_bn_in_mlp else nn.LayerNorm, eps=1e-6)
        self.norm = last_norm(embedding_dim)

    def load_state_dict(self, state_dict, strict=True):
        """ interpolate positional embedding if model is loaded from checkpoint and has different input resolution """
        old_pos_embed = state_dict["pos_embed"]
        # LEGACY: old checkpoints have pos_embed that stores zeros for cls token
        old_n_positions = old_pos_embed.shape[1]
        if int(old_n_positions ** 0.5) ** 2 + 1 == old_n_positions and self.cls_token is not None:
            if not torch.all(torch.zeros_like(old_pos_embed[:, 0]) == old_pos_embed[:, 0]):
                # public checkpoints can have learnable pos_embed for cls_token -> directly integrate into cls_token
                state_dict["cls_token"] += old_pos_embed[:, :1]
            old_pos_embed = old_pos_embed[:, 1:]
            state_dict["pos_embed"] = old_pos_embed
        # LEGACY: end
        if old_pos_embed.shape != self.pos_embed.shape:
            new_pos_embed = interpolate_pos_embed_permanent(self, old_pos_embed)
            state_dict["pos_embed"] = new_pos_embed
        super().load_state_dict(state_dict=state_dict, strict=strict)

    @property
    def _requires_initializer(self):
        return False

    def _model_specific_initialization(self):
        if self.use_learnable_pos_embed:
            # learnable pos_embed
            torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        else:
            # initialize pos_embed with sin-cos embedding
            img_height, img_width = self.input_shape[1:]
            h_seqlen, w_seqlen = get_sequence_lengths(self.patch_size, img_height, img_width)
            pos_embed = get_2d_sincos_pos_embed(self.embedding_dim, h_seqlen, w_seqlen)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        if isinstance(self.patch_embed, PatchEmbed):
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        elif isinstance(self.patch_embed, NonsharedPatchEmbed):
            # NonsharedPatchEmbed has nn.Linear projection -> is initialized anyways in self._init_weights
            # keep this here in case self._init_weights changes at some point
            for proj in self.patch_embed.proj:
                torch.nn.init.xavier_uniform_(proj.weight.data)
        # original impl doesnt initialize bias to zero
        # LEGACY: it should though (e.g. MoCoV3 uses it and it is inconsistent anyways)
        # torch.nn.init.zeros_(self.patch_embed.proj.bias)

        # class token
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(initialize_layernorm_as_noaffine)
        self.apply(initialize_linear_bias_to_zero)
        self.apply(self._init_weights)

        # LEGACY: this is not used by default because old runs didn't use it
        # https://github.com/facebookresearch/moco-v3/blob/main/vits.py#L35
        # for name, module in self.named_modules():
        #     if "qkv" in name:
        #         # treat the weights of Q, K, V separately
        #         val = (6. / float(module.weight.shape[0] // 3 + module.weight.shape[1])) ** 0.5
        #         nn.init.uniform_(module.weight, -val, val)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)

    def embed_and_add_pos(self, x):
        # interpolate pos embed if multiple resolutions are needed during training (e.g. multi-crop ssl)
        if self.input_shape != x.shape[1:]:
            _, old_h, old_w = self.input_shape
            _, _, new_h, new_w = x.shape
            pos_embed = interpolate_pos_embed_temporary(
                old_pos_embed=self.pos_embed,
                old_token_h=int(old_h / self.patch_embed.patch_size[0]),
                old_token_w=int(old_w / self.patch_embed.patch_size[1]),
                new_token_h=int(new_h / self.patch_embed.patch_size[0]),
                new_token_w=int(new_w / self.patch_embed.patch_size[1]),
            )
        else:
            pos_embed = self.pos_embed

        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = x + pos_embed
        x = self.pos_drop(x)
        return x

    def apply_transformer_blocks(self, x):
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = self._before_block(x, i)
            x = blk(x)
        x = self.norm(x)
        return x

    def append_cls_token(self, x):
        if self.cls_token is not None:
            cls_token = einops.repeat(self.cls_token, "1 n_tokens dim -> bs n_tokens dim", bs=len(x))
            x = torch.cat((cls_token, x), dim=1)
        return x

    # noinspection PyUnusedLocal
    @staticmethod
    def _before_block(x, idx):
        return x

    def forward(self, x):
        x = self.embed_and_add_pos(x)
        x = self.append_cls_token(x)
        x = self.apply_transformer_blocks(x)
        return x

    def features(self, x, pool_kind=None):
        # use VitMae here as subclasses can overwrite the forward pass for their training (e.g. MaskedEncoder)
        all_tokens = VitMae.forward(self, x)
        return SinglePooling.get_pool_fn(kind=pool_kind, model=self)(all_tokens)
