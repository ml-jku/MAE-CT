import kappaprofiler as kp
from kappadata import LabelSmoothingWrapper
from losses.soft_target_cross_entropy_loss import soft_target_cross_entropy_loss
from torch.nn.functional import cross_entropy

from losses import loss_fn_from_kwargs
from losses.bce_loss import bce_loss
from utils.factory import create
from utils.object_from_kwargs import objects_from_kwargs
from .base.sgd_trainer import SgdTrainer


class ClassificationTrainer(SgdTrainer):
    def __init__(self, forward_kwargs=None, loss_function=None, **kwargs):
        super().__init__(**kwargs)
        self._loss_function = create(loss_function, loss_fn_from_kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)

    @property
    def loss_function(self):
        if self._loss_function is None:
            ds = self.data_container.get_dataset("train")
            if ds.has_wrapper_type(LabelSmoothingWrapper):
                self._loss_function = soft_target_cross_entropy_loss
            elif ds.n_classes > 1 and not ds.is_multiclass:
                self._loss_function = cross_entropy
            else:
                self._loss_function = bce_loss
        return self._loss_function

    @property
    def output_shape(self):
        return self.data_container.get_dataset("train", mode=self.dataset_mode).n_classes,

    @property
    def dataset_mode(self):
        return "index x class"

    def forward(self, model, batch, train_dataset):
        (idx, x, y), ctx = batch
        with kp.named_profile_async("to_device"):
            x = x.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)
        with kp.named_profile_async("forward"):
            predictions = model(x, **self.forward_kwargs)
        # wrap model output into a dictionary in case it isn't already
        if not isinstance(predictions, dict):
            predictions = dict(main=predictions)
        return dict(predictions=predictions, x=x, y=y, idx=idx, **{f"ctx.{k}": v for k, v in ctx.items()})

    def get_loss(self, outputs, model):
        predictions = outputs["predictions"]
        classes = outputs["y"]

        with kp.named_profile_async("loss"):
            losses = {name: self.loss_function(prediction, classes) for name, prediction in predictions.items()}
            losses["total"] = sum(losses.values())
            unreduced_losses = {
                name: self.loss_function(prediction, classes, reduction="none")
                for name, prediction in predictions.items()
            }
            unreduced_losses["total"] = sum(unreduced_losses.values())

        return losses, dict(unreduced_losses=unreduced_losses, classes=classes, **outputs)
