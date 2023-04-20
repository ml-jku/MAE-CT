import torch.nn

from .functional.color_histogram_losses import color_histogram_regression_loss, color_histogram_ce_loss
from torch.nn.functional import l1_loss
from functools import partial
from .base.color_histogram_loss import ColorHistogramLoss
from utils.factory import create
from losses import loss_fn_from_kwargs


class ColorHistogramRegressionLoss(ColorHistogramLoss):
    def __init__(self, loss_function, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = create(loss_function, loss_fn_from_kwargs)

    @property
    def loss_fn(self):
        return partial(color_histogram_regression_loss, loss_fn=l1_loss)


