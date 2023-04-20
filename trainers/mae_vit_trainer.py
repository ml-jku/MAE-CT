import kappaprofiler as kp
import torch.nn as nn

from losses import loss_fn_from_kwargs
from losses.mae_vit_loss import mae_vit_loss
from models.vit.mask_generators import mask_generator_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class MaeVitTrainer(SgdTrainer):
    def __init__(self, mask_generator, normalize_pixels, loss_function=None, **kwargs):
        super().__init__(**kwargs)
        self.mask_generator = create(mask_generator, mask_generator_from_kwargs, update_counter=self.update_counter)
        self.normalize_pixels = normalize_pixels
        if loss_function is not None:
            self.loss_function = create(loss_function, loss_fn_from_kwargs, reduction="none")
        else:
            self.loss_function = nn.MSELoss(reduction="none")

    @property
    def output_shape(self):
        # not required as MaskedAutoencoderVit sets its own output_shape
        return None

    @property
    def dataset_mode(self):
        return "x"

    def forward(self, model, batch, train_dataset, mask_generator=None):
        x, _ = batch
        with kp.named_profile_async("to_device"):
            x = x.to(model.device, non_blocking=True)
        if x.ndim == 5:
            x = train_dataset.to_concat_view(x)

        # for calculating the loss for logging, a mask generator has to be provided in order to be deterministic
        mask_generator = mask_generator or self.mask_generator

        with kp.named_profile_async("forward"):
            outputs = model(x, mask_generator=mask_generator)
        outputs["x"] = x
        return outputs

    def get_loss(self, outputs, model):
        mask = outputs["mask"]
        if "x_hat" not in outputs:
            return dict(total=0.), dict(mask=mask)

        x_on_device = outputs["x"]
        x_hat = outputs["x_hat"]

        with kp.named_profile_async("loss"):
            loss = mae_vit_loss(
                x=x_on_device,
                x_hat=x_hat,
                patch_size=model.encoder.patch_size,
                normalize_pixels=self.normalize_pixels,
                base_loss_function=self.loss_function,
                mask=mask,
            )
        mask_ratio = mask.sum() / mask.numel()
        return dict(total=loss, reconstruction=loss), dict(x_hat=x_hat, mask=mask, mask_ratio=mask_ratio)
