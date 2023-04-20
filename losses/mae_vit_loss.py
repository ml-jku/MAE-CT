import torch.nn as nn
import torch.nn.functional as F

from losses import loss_fn_from_kwargs
from utils.factory import create
from utils.vit_util import patchify_as_1d


class MaeVitLoss(nn.Module):
    def __init__(
            self,
            normalize_pixels,
            loss_function=None,
            include_visible_patches=False,
            weight_global_as_token=True,
    ):
        super().__init__()
        self.normalize_pixels = normalize_pixels
        self.include_visible_patches = include_visible_patches
        self.weight_global_as_token = weight_global_as_token
        self.loss_function = create(loss_function, loss_fn_from_kwargs, reduction="none")
        if self.loss_function is None:
            self.loss_function = nn.MSELoss(reduction="none")
        assert self.loss_function.reduction == "none"

    def forward(self, prediction, target, mask, patch_size):
        # multi-view case
        if isinstance(prediction, list):
            return {
                f"view{i}": self(*args, patch_size)
                for i, args in enumerate(zip(prediction, target, mask))
            }

        patches_prediction = prediction["patches"]

        # prepare target
        patchified_target = patchify_as_1d(imgs=target, patch_size=patch_size)
        # normalize reconstructed pixels
        if self.normalize_pixels:
            mean = patchified_target.mean(dim=-1, keepdim=True)
            var = patchified_target.var(dim=-1, keepdim=True)
            patchified_target = (patchified_target - mean) / (var + 1.e-6) ** .5

        # unreduced loss
        patch_loss = self.loss_function(patches_prediction, patchified_target)
        # [batch_size, n_patches, c*prod(patch_size))] -> [batch_size, n_patches] (mean loss per patch)
        patch_loss = patch_loss.mean(dim=-1)
        if self.include_visible_patches:
            patch_loss = patch_loss.mean()
        else:
            # mean loss on removed patches (mask is 1 if the patch was removed)
            patch_loss = (patch_loss * mask).sum() / mask.sum()

        # decoder predicts full image (downsampled) from cls token
        if "global" in prediction:
            global_prediction = prediction["global"]
            global_size = int((global_prediction.shape[2] // target.shape[1]) ** 0.5)
            global_target = patchify_as_1d(
                imgs=F.interpolate(target, size=global_size, mode='bicubic'),
                patch_size=global_size,
            )
            if self.normalize_pixels:
                mean = global_target.mean(dim=-1, keepdim=True)
                var = global_target.var(dim=-1, keepdim=True)
                global_target = (global_target - mean) / (var + 1.e-6) ** .5
            global_loss = self.loss_function(global_prediction, global_target).mean()
            if self.weight_global_as_token:
                n_loss_tokens = len(mask[0]) if self.include_visible_patches else mask[0].sum()
                global_loss = global_loss / n_loss_tokens
        else:
            global_loss = 0

        loss = patch_loss + global_loss
        return loss


# TODO only needed for old contrastive impl
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
