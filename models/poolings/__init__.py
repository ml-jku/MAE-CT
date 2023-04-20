import torch.nn as nn

from models.poolings.single_pooling import SinglePooling
from utils.factory import instantiate
from copy import deepcopy

class IdentityPooling:
    def __call__(self, x, ctx):
        return x

# TODO not sure why here pooling is required instead of kind
def pooling_from_kwargs(pooling, **kwargs):
    if pooling is None:
        return IdentityPooling()
    if not isinstance(pooling, dict):
        return pooling
    # TODO a cleaner solution is to make every pooling entity a seperate instance
    pooling = deepcopy(pooling)
    kind = pooling.pop("kind")
    if kind.startswith("extractor"):
        return instantiate(module_names=[f"models.poolings.{kind}"], type_names=[kind], **pooling, **kwargs)
    return SinglePooling(kind, **pooling, **kwargs)



def pooling2d_from_kwargs(kind, factor):
    if kind is None:
        return nn.Identity()
    if kind == "max":
        return nn.MaxPool2d(kernel_size=factor, stride=factor)
    if kind in ["mean", "avg", "average"]:
        return nn.AvgPool2d(kernel_size=factor, stride=factor)
    raise NotImplementedError
