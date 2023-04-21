import einops
import torch
from datasets.transforms import transform_from_kwargs, transform_collate_fn
from kappadata import KDWrapper

from utils.factory import create_collection


class MultiViewWrapper(KDWrapper):
    def __init__(self, n_views=None, transforms=None, **kwargs):
        super().__init__(**kwargs)
        # n_views is not None -> sample the same sample n_views times from the dataset
        # transforms is not None -> sample once from dataset then apply each transform to the same sample
        assert (n_views is None) ^ (transforms is None)
        self._n_views = n_views
        if transforms is not None:
            self.transforms = [
                create_collection(transform, transform_from_kwargs, collate_fn=transform_collate_fn)
                for transform in transforms
            ]
        else:
            self.transforms = None

    @property
    def n_views(self):
        if self._n_views is not None:
            return self._n_views
        return len(self.transforms)

    def getitem_x(self, idx, ctx=None):
        if self._n_views is not None:
            x = []
            for i in range(self._n_views):
                cur_ctx = {}
                x.append(self.dataset.getitem_x(idx, cur_ctx))
                if ctx is not None:
                    ctx[f"view{i}"] = cur_ctx
        else:
            sample = self.dataset.getitem_x(idx)
            x = [transform(sample, ctx) for i, transform in enumerate(self.transforms)]
        return torch.stack(x)

    @staticmethod
    def to_concat_view(x):
        """
        transform [batch_size, n_views, ...] to [batch_size * n_views, ...]
        [:1*batch_size, ...] is view1
        [:2*batch_size, ...] is view2
        ...
        """
        return einops.rearrange(x, "b v ... -> (v b) ...")

    def to_split_view(self, x):
        """ transform [batch_size * n_views, ...] to [batch_size, n_views, ...] """
        return einops.rearrange(x, "(v b) ... -> b v ...", v=self.n_views)
