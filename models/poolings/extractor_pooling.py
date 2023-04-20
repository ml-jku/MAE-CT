import torch
from .base.pooling_base import PoolingBase
from utils.factory import create
from models.poolings import pooling_from_kwargs

class ExtractorPooling(PoolingBase):
    def __init__(self, keys, pooling=None, **kwargs):
        super().__init__(**kwargs)
        self.keys = keys
        self.pooling = pooling_from_kwargs(pooling)

    def forward(self, x, ctx, *_, **__):
        features = []
        for key in self.keys:
            feature = ctx[key]
            pooled = self.pooling(feature, ctx=ctx)
            features.append(pooled)
        return torch.concat(features, dim=1)


