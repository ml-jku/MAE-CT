import torch


class ConcatFinalizer:
    def __call__(self, features):
        return torch.concat(features)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__
