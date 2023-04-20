import numpy as np
import torch
import torch.nn as nn

from models.base.single_model_base import SingleModelBase
from models.poolings import pooling_from_kwargs
from utils.factory import create
from initializers.trunc_normal_initializer import TruncNormalInitializer


class LinearHead(SingleModelBase):
    def __init__(self, nonaffine_batchnorm=False, pooling=None, initializer=None, **kwargs):
        initializer = initializer or TruncNormalInitializer(std=1e-2)
        super().__init__(initializer=initializer, **kwargs)
        self.nonaffine_batchnorm = nonaffine_batchnorm
        self.pooling = pooling_from_kwargs(pooling)
        input_shape = self.pooling(torch.ones(1, *self.input_shape), ctx=self.ctx).shape[1:]
        input_dim = np.prod(input_shape)
        self.norm = nn.BatchNorm1d(input_dim, affine=False) if nonaffine_batchnorm else nn.Identity()
        self.layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            self.norm,
            nn.Linear(input_dim, np.prod(self.output_shape)),
        )
        # normalization values that can be patched by PrepareFeatureStatisticsLogger
        self.mean = None
        self.std = None

    def load_state_dict(self, state_dict, strict=True):
        # previously the pooling was within the sequential so indices are messed up
        if "layer.3.weight" in state_dict:
            # "layer.1.running_mean", "layer.1.running_var", "layer.2.weight", "layer.2.bias"
            state_dict["layer.1.running_mean"] = state_dict["layer.2.running_mean"]
            state_dict["layer.1.running_var"] = state_dict["layer.2.running_var"]
            state_dict["layer.1.num_batches_tracked"] = state_dict["layer.2.num_batches_tracked"]
            state_dict["layer.2.weight"] = state_dict["layer.3.weight"]
            state_dict["layer.2.bias"] = state_dict["layer.3.bias"]
            del state_dict["layer.2.running_mean"]
            del state_dict["layer.2.running_var"]
            del state_dict["layer.2.num_batches_tracked"]
            del state_dict["layer.3.weight"]
            del state_dict["layer.3.bias"]
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, x):
        x = self.pooling(x, ctx=self.ctx)
        if self.mean is not None and self.std is not None:
            assert isinstance(self.norm, nn.Identity)
            x = (x - self.mean) / self.std
        return self.layer(x)

    def features(self, x):
        return self(x)

    def predict(self, x):
        return dict(main=self(x))

    def predict_binary(self, x):
        return self.predict(x)