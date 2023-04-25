import torch.nn as nn
import torch.nn.functional as F

from losses import loss_fn_from_kwargs
from utils.factory import create
from utils.vit_util import patchify_as_1d


def mae_vit_loss(x, x_hat, patch_size, normalize_pixels, base_loss_function, mask):
    patchified_x = patchify_as_1d(imgs=x, patch_size=patch_size)

    # normalize reconstructed pixels
    if normalize_pixels:
        mean = patchified_x.mean(dim=-1, keepdim=True)
        var = patchified_x.var(dim=-1, keepdim=True)
        patchified_x = (patchified_x - mean) / (var + 1.e-6) ** .5
    # unreduced loss
    loss = base_loss_function(x_hat, patchified_x)
    # [batch_size, n_patches, c*prod(patch_size))] -> [batch_size, n_patches] (mean loss per patch)
    loss = loss.mean(dim=-1)
    # mean loss on removed patches (mask is 1 if the patch was removed)
    # loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
    # TODO
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss
