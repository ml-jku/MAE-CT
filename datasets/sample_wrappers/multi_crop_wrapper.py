from kappadata import KDWrapper

from datasets.transforms import transform_from_kwargs, transform_collate_fn
from utils.factory import create_collection


class MultiCropWrapper(KDWrapper):
    def __init__(self, transform_configs, **kwargs):
        super().__init__(**kwargs)
        self.views_per_transform = []
        self.transforms = []
        for transform_config in transform_configs:
            self.views_per_transform.append(transform_config.get("n_views", 1))
            self.transforms.append(
                create_collection(
                    transform_config["transforms"],
                    transform_from_kwargs,
                    collate_fn=transform_collate_fn,
                ),
            )
        self.n_views = sum(self.views_per_transform)

    def getitem_x(self, idx, ctx=None):
        sample = self.dataset.getitem_x(idx)

        x = []
        counter = 0
        for n_views, transform in zip(self.views_per_transform, self.transforms):
            for _ in range(n_views):
                view_ctx = {}
                view = transform(sample, view_ctx)
                x.append(view)
                if ctx is not None:
                    ctx[f"view{counter}"] = view_ctx
                    counter += 1
        return x
