import einops
import torch

from .param_checking import to_2tuple


def get_sequence_lengths(patch_size, img_height, img_width):
    patch_height, patch_width = to_2tuple(patch_size)
    h_seqlen = img_height // patch_height
    w_seqlen = img_width // patch_width
    return h_seqlen, w_seqlen


def sequence_to_2d_with_seqlens(tokens, h_seqlen, w_seqlen, n_aux_tokens):
    # transform into image with c=feature_dim h=h_seqlen w=w_seqlen
    aux_tokens = tokens[:, :n_aux_tokens]
    patch_tokens = tokens[:, n_aux_tokens:]
    img = einops.rearrange(
        patch_tokens,
        "b (h_seqlen w_seqlen) c -> b c h_seqlen w_seqlen",
        h_seqlen=h_seqlen,
        w_seqlen=w_seqlen,
    )
    return img, aux_tokens


def flatten_2d_to_1d(patch_tokens_as_img, aux_tokens):
    patch_tokens = einops.rearrange(patch_tokens_as_img, "b c h_seqlen w_seqlen -> b (h_seqlen w_seqlen) c")
    if aux_tokens is not None and len(aux_tokens) > 0:
        return torch.cat([aux_tokens, patch_tokens], dim=1)
    return patch_tokens


def patchify_as_1d(imgs, patch_size):
    patch_height, patch_width = to_2tuple(patch_size)
    bs, c, img_h, img_w = imgs.shape
    assert img_h == img_w and img_h % patch_height == 0 and img_w % patch_width == 0
    # how many patches are along height/width dimension
    h = img_h // patch_height
    w = img_w // patch_width
    # return as sequence
    x = einops.rearrange(imgs, "bs c (h ph) (w pw) -> bs (h w) (ph pw c)", h=h, ph=patch_height, w=w, pw=patch_width)
    return x


def unpatchify_from_1d(patches, patch_size, img_shape=None):
    assert patches.ndim == 3
    patch_height, patch_width = to_2tuple(patch_size)
    assert patch_height == patch_width or img_shape is not None
    if img_shape is not None:
        # derive number of patches along height/width from original image shape
        _, img_h, img_w = img_shape
        assert img_h % patch_height == 0 and img_w % patch_width == 0
        h = img_h // patch_height
        w = img_w // patch_width
    else:
        # equal number of patches along height/width
        h = w = int(patches.shape[1] ** .5)
    return einops.rearrange(
        patches,
        "bs (h w) (ph pw c) -> bs c (h ph) (w pw)",
        ph=patch_height,
        pw=patch_width,
        h=h,
        w=w,
    )


def patchify_as_2d(imgs, patch_size):
    patch_height, patch_width = to_2tuple(patch_size)
    bs, c, img_h, img_w = imgs.shape
    assert img_h == img_w and img_h % patch_height == 0 and img_w % patch_width == 0
    # how many patches are along height/width dimension
    h = img_h // patch_height
    w = img_w // patch_width
    # return as "image"
    x = einops.rearrange(imgs, "bs c (h ph) (w pw) -> bs (ph pw c) h w", h=h, ph=patch_height, w=w, pw=patch_width)
    return x


def unpatchify_from_2d(patches, patch_size):
    patch_height, patch_width = to_2tuple(patch_size)
    return einops.rearrange(patches, "bs (ph pw c) h w -> bs c (h ph) (w pw)", ph=patch_height, pw=patch_width)
