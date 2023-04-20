import einops
import torch
from torch.nn.functional import mse_loss
from kappadata import color_histogram
from functools import partial


def color_histogram_regression_loss(preds, images, bins, temperature=1., loss_fn=mse_loss, reduction="mean"):
    channels = images.shape[1]
    if preds.ndim != 3:
        preds = einops.rearrange(preds, "bs (bins channels) -> bs channels bins", channels=channels, bins=bins)
    if temperature is None:
        # preds should be a distribution -> check if sums to 1
        sums = preds.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums)), sums
    else:
        # make preds into a distribution
        preds = (preds / temperature).softmax(dim=2)
    densities = color_histogram(images, bins=bins, density=True, batch_size=128)
    return loss_fn(preds, densities, reduction=reduction)