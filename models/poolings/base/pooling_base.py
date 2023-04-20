import torch.nn as nn


# derive from nn.Module to be usable in nn.Sequential
class PoolingBase(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        if model is not None:
            raise NotImplementedError(
                "if model is passed to pooling, all parameters automatically become part of e.g. a NnclrHead "
                "which leads to the parameters of the backbone being updated twice (once with the optim from "
                "NnclrHead and once with the actual optim of the backbone)"
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError
