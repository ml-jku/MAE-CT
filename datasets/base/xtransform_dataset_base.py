from datasets.transforms import transform_from_kwargs, transform_collate_fn
from utils.factory import create_collection
from .dataset_base import DatasetBase


class XTransformDatasetBase(DatasetBase):
    def __init__(self, x_transform=None, **kwargs):
        super().__init__(**kwargs)
        self.x_transform = create_collection(x_transform, transform_from_kwargs, collate_fn=transform_collate_fn)

    def __len__(self):
        raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        raise NotImplementedError
