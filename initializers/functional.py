import torch.nn as nn

ALL_BATCHNORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
)

_ALL_NORMS = (
    *ALL_BATCHNORMS,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LocalResponseNorm,
)


def initialize_norms_as_noaffine(m):
    if isinstance(m, _ALL_NORMS):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 1.)


def initialize_norms_as_identity(m):
    if isinstance(m, _ALL_NORMS):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def initialize_layernorm_as_noaffine(m):
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 1.)


def initialize_layernorm_as_identity(m):
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def initialize_batchnorm_as_noaffine(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 1.)


def initialize_batchnorm_as_identity(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.bias, 0.)
        nn.init.constant_(m.weight, 0.)
    else:
        raise NotImplementedError


def initialize_linear_bias_to_zero(m):
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
