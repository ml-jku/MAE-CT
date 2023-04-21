import kappaprofiler as kp
import torch

from models.vit.mask_generators.random_mask_generator import RandomMaskGenerator
from utils.object_from_kwargs import objects_from_kwargs
from .base.sgd_trainer import SgdTrainer


class MaeContheadsVitForwardTrainer(SgdTrainer):
    def __init__(self, forward_kwargs=None, disable_backward=None, **kwargs):
        assert disable_backward or disable_backward is None
        super().__init__(disable_backward=True, **kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)

    @property
    def output_shape(self):
        return None

    @property
    def dataset_mode(self):
        return "index x class"

    def forward(self, model, batch, train_dataset):
        (idx, x, y), ctx = batch
        with kp.named_profile_async("to_device"):
            x = x.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        with kp.named_profile_async("forward"):
            model(x, mask_generator=RandomMaskGenerator(mask_ratio=0.), batch_size=len(y), **self.forward_kwargs)
        return dict(x=x, classes=y, idx=idx, **{f"ctx.{k}": v for k, v in ctx.items()})

    def get_loss(self, outputs, model):
        return dict(total=torch.tensor(0.)), outputs
