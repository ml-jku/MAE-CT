from functools import partial

import kappaprofiler as kp
import torch

from kappadata import KDMultiViewWrapper
from datasets.sample_wrappers.multi_view_wrapper import MultiViewWrapper
from datasets.transforms import transform_from_kwargs, transform_collate_fn
from schedules import schedule_from_kwargs
from utils.factory import create_collection
from .mae_vit_trainer import MaeVitTrainer


class MaeContheadsVitTrainer(MaeVitTrainer):
    def __init__(self, transforms=None, transforms_schedule=None, **kwargs):
        super().__init__(**kwargs)
        if transforms is not None:
            self.transforms = [
                create_collection(transform, transform_from_kwargs, collate_fn=transform_collate_fn)
                for transform in transforms
            ]
        else:
            self.transforms = None
        self.transforms_schedule = schedule_from_kwargs(transforms_schedule, update_counter=self.update_counter)

    @property
    def dataset_mode(self):
        return "index x class"

    def forward(self, model, batch, train_dataset, mask_generator=None):
        outputs = {}
        (idx, x, y), ctx = batch



        # patch KDMultiViewWrapper
        if isinstance(x, list):
            train_dataset.n_views = len(x)
            train_dataset.to_concat_view = MultiViewWrapper.to_concat_view
            train_dataset.to_split_view = partial(MultiViewWrapper.to_split_view, train_dataset)
            # push first to GPU...stacking on CPU takes longer
            with kp.named_profile_async("to_device"):
                x = [item.to(model.device, non_blocking=True) for item in x]
            x = torch.stack(x, dim=1)
        else:
            with kp.named_profile_async("to_device"):
                x = x.to(model.device, non_blocking=True)
        idx = idx.to(model.device, non_blocking=True)
        y = y.to(model.device, non_blocking=True)

        # augmentation warmup
        if self.transforms is not None:
            assert x.ndim == 5
            # scale augmentation strength
            if self.transforms_schedule is not None:
                scale = self.transforms_schedule.get_value(self.update_counter.cur_checkpoint)
                for transform in self.transforms:
                    transform.scale_strength(scale)
                outputs["transform_scale"] = scale

            samples = []
            # use float32 to avoid "RuntimeError: "reflection_pad2d_out_template" not implemented for 'BFloat16'"
            with torch.autocast(str(model.device).replace(":0", ""), dtype=torch.float32):
                for sample in x:
                    samples.append(torch.stack([transform(view) for view, transform in zip(sample, self.transforms)]))
                x = torch.stack(samples)
            # patch MultiViewWrapper properties
            train_dataset.n_views = len(self.transforms)
            train_dataset.to_concat_view = MultiViewWrapper.to_concat_view
            train_dataset.to_split_view = partial(MultiViewWrapper.to_split_view, train_dataset)

        # get batch_size (x.shape is [batch_size, n_views, ...]
        batch_size = len(x)

        # change x from [batch_size, n_views, ...] -> [n_views * batch_size, ...]
        if x.ndim == 5:
            x = MultiViewWrapper.to_concat_view(x)

        # for calculating the loss for logging, a mask generator has to be provided in order to be deterministic
        mask_generator = mask_generator or self.mask_generator

        with kp.named_profile_async("forward"):
            outputs.update(model(x, mask_generator=mask_generator, batch_size=batch_size))
        outputs["x"] = x
        outputs["idx"] = idx
        outputs["y"] = y
        outputs["mask_ratio"] = mask_generator.mask_ratio
        if "view0" in ctx.keys():
            for view_name in ctx.keys():
                outputs.update({f"ctx.{view_name}.{k}": v for k, v in ctx[view_name].items()})
        return outputs

    def get_loss(self, outputs, model):
        mae_losses, mae_outputs = super().get_loss(outputs, model)
        mae_total_loss = mae_losses.pop("total")

        idx = outputs["idx"]
        y = outputs["y"]

        all_head_losses = {}
        all_head_outputs = {}
        all_total_losses = {}
        for head_name, head in model.contrastive_heads.items():
            head_losses, head_outputs = head.get_loss(outputs[head_name], idx=idx, y=y)
            all_total_losses[head_name] = head_losses.pop("total")
            for loss_name, head_loss in head_losses.items():
                all_head_losses[f"{head_name}/{loss_name}"] = head_loss
            for output_name, head_output in head_outputs.items():
                all_head_outputs[f"{head_name}/{output_name}"] = head_output

        total_loss = mae_total_loss + sum(all_total_losses.values())
        loss_outputs = dict(**mae_outputs, **all_head_outputs, latent_tokens=outputs["latent_tokens"])
        loss_outputs["mask_ratio"] = outputs["mask_ratio"]
        loss_outputs["classes"] = y
        return dict(total=total_loss, **mae_losses, **all_head_losses), {**loss_outputs, **outputs}
