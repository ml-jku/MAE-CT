import torch

class ColorHistogramLoss(torch.nn.Module):
    def __init__(self, bins: int, temperature: float = 1., reduction="mean"):
        super().__init__()
        self.bins = bins
        self.temperature = temperature
        self.reduction = reduction

    @property
    def loss_fn(self):
        raise NotImplementedError

    def forward(self, preds, images):
        return self.loss_fn(
            preds=preds,
            images=images,
            bins=self.bins,
            temperature=self.temperature,
            reduction=self.reduction,
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}(bins={self.bins},temp={self.temperature})"