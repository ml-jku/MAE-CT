import torch

from .base.dataset_base import DatasetBase


class DummyDataset(DatasetBase):
    def __init__(self, x_shape, size=None, n_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.x_shape = x_shape
        self._n_classes = n_classes

    def __len__(self):
        # return a large value divisible by 2 to avoid specifying a size when the dataset is only used
        # for the eval_trainer to know the input shapes
        return self.size or 65536

    def getitem_x(self, idx, ctx=None):
        return torch.randn(*self.x_shape, generator=torch.Generator().manual_seed(int(idx)))

    @property
    def n_classes(self):
        return self._n_classes if self._n_classes > 2 else 1

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        return torch.randint(0, self.n_classes, size=(1,), generator=torch.Generator().manual_seed(int(idx))).item()
