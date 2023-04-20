import torch
from .base.pooling_base import PoolingBase
from utils.factory import create
from models.poolings import pooling_from_kwargs

class ExtractorLastMeanPooling(PoolingBase):
    """
    helper class to avoid specifying keys in the poolings
    this class will always use the last n_keys from the extractor
    by using this it can be avoided to write seperate yamls for ViT-B and ViT-L
    """
    def __init__(self, n_keys, pooling=None, **kwargs):
        super().__init__(**kwargs)
        self.n_keys = n_keys
        self.pooling = pooling_from_kwargs(pooling)

    def forward(self, x, ctx, *_, **__):
        features = []
        for key in list(ctx.keys())[-self.n_keys:]:
            feature = ctx[key]
            pooled = self.pooling(feature, ctx=ctx)
            features.append(pooled)
        return torch.mean(torch.stack(features), dim=0)