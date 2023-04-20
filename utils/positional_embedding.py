# adapted from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
import logging
import math

import einops
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, h_seqlen, w_seqlen):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(h_seqlen, dtype=float)
    grid_w = np.arange(w_seqlen, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h_seqlen, w_seqlen])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_continuous(x, embed_dim: int):
    """ https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py """
    assert x.ndim == 1
    half_dim = embed_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
    emb = torch.einsum("m,d->md", x, emb)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, seqlen):
    grid = np.arange(seqlen, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed_permanent(model, old_pos_embed):
    assert model.patch_embed.patch_size[0] == model.patch_embed.patch_size[1], "only square patchsizes supported"
    _, img_h, img_w = model.input_shape
    assert img_h == img_w, "only square patchsizes supported"

    _, old_n_patches, dim = old_pos_embed.shape
    new_n_patches = model.patch_embed.num_patches

    old_size = int(old_n_patches ** 0.5)
    new_size = int(new_n_patches ** 0.5)

    if old_size == new_size:
        return old_pos_embed

    logging.info(f"position embedding interpolated from {old_size}x{old_size} to {new_size}x{new_size}")
    # aux_tokens are kept unchanged
    new_pos_embed = F.interpolate(
        einops.rearrange(old_pos_embed, "1 (h w) dim -> 1 dim h w", h=old_size, w=old_size),
        size=(new_size, new_size),
        mode='bicubic',
    )
    new_pos_embed = einops.rearrange(new_pos_embed, "1 dim h w -> 1 (h w) dim")
    return new_pos_embed


# interpolate positional embedding only for the current forward pass
def interpolate_pos_embed_temporary(old_pos_embed, old_token_h, old_token_w, new_token_h, new_token_w):
    new_pos_embed = F.interpolate(
        einops.rearrange(old_pos_embed, "1 (h w) dim -> 1 dim h w", h=old_token_h, w=old_token_w),
        size=(new_token_h, new_token_w),
        mode='bicubic',
    )
    new_pos_embed = einops.rearrange(new_pos_embed, "1 dim h w -> 1 (h w) dim")
    return new_pos_embed
