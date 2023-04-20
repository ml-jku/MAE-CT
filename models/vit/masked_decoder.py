from functools import partial

import einops
import torch
import torch.nn as nn

from initializers.functional import initialize_linear_bias_to_zero, initialize_layernorm_as_noaffine
from models.modules.vit_block import VitBlock
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create
from utils.param_checking import to_2tuple
from utils.positional_embedding import get_2d_sincos_pos_embed
from utils.vit_util import get_sequence_lengths
from ..base.single_model_base import SingleModelBase


class MaskedDecoder(SingleModelBase):
    def __init__(
            self,
            patch_size, 
            embedding_dim, 
            depth, 
            attention_heads, 
            n_aux_tokens, 
            mask_generator=None,
            use_aux_tokens=True,
            detach_aux_tokens=False,
            detach=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = to_2tuple(patch_size)
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.attention_heads = attention_heads
        self.n_aux_tokens = n_aux_tokens
        # for experiment that has encoder without mask and does the masking in the decoder to still have a task
        self.mask_generator = create(mask_generator, mask_generator_from_kwargs, update_counter=self.update_counter)
        self.use_aux_tokens = use_aux_tokens
        self.detach_aux_tokens = detach_aux_tokens
        self.detach = detach

        # decoder doesn't produce original image shape but flattened patches
        flat_patch_dim = self.patch_size[0] * self.patch_size[1] * self.output_shape[0]
        num_tokens, encoder_embedding_dim = self.input_shape
        # remember original output_shape (e.g. for unpatchify)
        self.original_output_shape = self.output_shape
        self.output_shape = (num_tokens, flat_patch_dim)

        self.embed = nn.Linear(encoder_embedding_dim, embedding_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        # fixed embedding
        self.register_buffer("pos_embed", torch.zeros(1, num_tokens - n_aux_tokens, embedding_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=embedding_dim,
                num_heads=attention_heads,
                qkv_bias=True,
                norm1_layer=norm_layer,
                norm2_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embedding_dim)
        # decoder to patch
        self.pred = nn.Linear(embedding_dim, flat_patch_dim, bias=True)

    def load_state_dict(self, state_dict, strict=True):
        old_pos_embed = state_dict["pos_embed"]
        # LEGACY: old checkpoints have pos_embed that stores zeros for cls token
        old_n_positions = old_pos_embed.shape[1]
        if int(old_n_positions ** 0.5) ** 2 + 1 == old_n_positions:
            state_dict["pos_embed"] = old_pos_embed[:, 1:]
        # LEGACY: end
        super().load_state_dict(state_dict=state_dict, strict=strict)

    @property
    def _requires_initializer(self):
        return False

    def _model_specific_initialization(self):
        # mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize pos_embed with sin-cos embedding
        _, img_height, img_width = self.original_output_shape
        h_seqlen, w_seqlen = get_sequence_lengths(self.patch_size, img_height, img_width)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.embedding_dim, h_seqlen, w_seqlen)
        self.pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.apply(initialize_layernorm_as_noaffine)
        self.apply(initialize_linear_bias_to_zero)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)

    # noinspection PyMethodOverriding
    def forward(self, x, ids_restore, mask=None, mask_generator=None, single_mask=False):
        if self.detach:
            x = x.detach()
        # for experiment that has encoder without mask and does the masking in the decoder to still have a task
        mask_generator = mask_generator or self.mask_generator
        if mask_generator is not None:
            aux_tokens = x[:, :self.n_aux_tokens]
            h = self.original_output_shape[1] // self.patch_size[0]
            w = self.original_output_shape[2] // self.patch_size[1]
            x = einops.rearrange(x[:, self.n_aux_tokens:], "bs (h w) dim -> bs dim h w", h=h, w=w)
            x, mask, ids_restore = mask_generator.get_mask(x, single_mask=single_mask)
            x = torch.concat([aux_tokens, x], dim=1)

        # embed tokens
        x = self.embed(x)

        # extract shapes
        bs, n_input_tokens, dim = x.shape
        _, total_n_patches = ids_restore.shape
        n_hidden_patches = total_n_patches - (n_input_tokens - self.n_aux_tokens)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(bs, n_hidden_patches, 1)
        # no aux tokens
        all_patches = torch.cat([x[:, self.n_aux_tokens:, :], mask_tokens], dim=1)
        # unshuffle
        indices = einops.repeat(ids_restore, "bs np -> bs np dim", dim=dim)
        all_patches = torch.gather(all_patches, dim=1, index=indices)
        # add pos embed
        all_patches = all_patches + self.pos_embed
        if self.use_aux_tokens:
            # append aux tokens
            if self.detach_aux_tokens:
                x = torch.cat([x[:, :self.n_aux_tokens, :].detach(), all_patches], dim=1)
            else:
                x = torch.cat([x[:, :self.n_aux_tokens, :], all_patches], dim=1)
        else:
            x = all_patches

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)
        
        if self.use_aux_tokens:
            # remove aux token
            x = x[:, self.n_aux_tokens:, :]

        if mask is not None:
            return x, mask
        return x
