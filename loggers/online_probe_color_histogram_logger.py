from collections import defaultdict
from functools import partial

import einops
import numpy as np
import torch
from kappadata import ModeWrapper
from kappadata import get_norm_transform

from distributed.gather import all_reduce_mean_grad
from loggers.base.dataset_logger import DatasetLogger
from losses import loss_fn_from_kwargs
from models import model_from_kwargs
from models.extractors import extractor_from_kwargs
from utils.factory import create
from utils.factory import create_collection
from utils.formatting_util import float_to_scientific_notation


class OnlineProbeColorHistogramLogger(DatasetLogger):
    def __init__(self, extractors, models, loss_function, **kwargs):
        super().__init__(**kwargs)
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        self.tracked_losses = defaultdict(list)
        self.models_kwargs = models
        self.loss_function = create(loss_function, loss_fn_from_kwargs, reduction="mean")
        self.loss_function_noreduce = create(loss_function, loss_fn_from_kwargs, reduction="none")
        self.model_keys = None
        self.models = None
        self.norm_transform = None

    @property
    def base_key(self):
        return f"color_hist_loss/{self.loss_function}"

    def get_dataset_mode(self, trainer):
        return trainer.dataset_mode

    def return_ctx(self):
        return True

    @property
    def allows_multiple_interval_types(self):
        return False

    def _before_training_impl(self, model, train_dataset, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)
        self.norm_transform = get_norm_transform(train_dataset.x_transform)

    def before_every_update(self, update_counter, **kwargs):
        for extractor in self.extractors:
            extractor.enable_hooks()
        # self.models is not initialized before first update step
        if self.models is not None:
            for model in self.models.values():
                model.optim.schedule_update_step(update_counter.cur_checkpoint)

    def _track_after_update_step(self, trainer, **kwargs):
        for model in self.models.values():
            model.optim.step(trainer.grad_scaler)
            model.optim.zero_grad()
        for extractor in self.extractors:
            extractor.disable_hooks()

    @staticmethod
    def _model_to_key(model, model_kwargs):
        opt_str = type(model.optim.torch_optim).__name__
        lr_str = float_to_scientific_notation(model_kwargs["optim"]["lr"], max_precision=0)
        sched_str = str(model.optim.schedule)
        return f"{type(model.unwrapped_ddp_module).__name__}(opt={opt_str},lr={lr_str},sched={sched_str})"

    def denormalize(self, x):
        pattern = "channels -> 1 channels 1 1"
        mean = einops.rearrange(torch.tensor(self.norm_transform.mean, device=x.device), pattern)
        std = einops.rearrange(torch.tensor(self.norm_transform.std, device=x.device), pattern)
        return (x * std + mean) * 255

    def _track_after_accumulation_step(
            self,
            trainer,
            model,
            accumulation_steps,
            update_outputs,
            train_dataset,
            update_counter,
            **kwargs,
    ):
        for x_key in ["x"]:
            if x_key in update_outputs:
                x = self.denormalize(update_outputs[x_key])
                break
        else:
            raise NotImplementedError
        assert x.ndim == 4
        all_features = [extractor.extract().detach() for extractor in self.extractors]

        if self.models is None:
            self.models = {}
            for extractor, features in zip(self.extractors, all_features):
                models = create_collection(
                    self.models_kwargs,
                    model_from_kwargs,
                    input_shape=features.shape[1:],
                    output_shape=(x.shape[1] * self.loss_function.bins,),
                    update_counter=update_counter,
                )
                for i in range(len(models)):
                    models[i] = models[i].to(model.device)
                    trainer.initialize_model(models[i])
                    models[i] = trainer.wrap_ddp(models[i])
                    models[i].optim.schedule_update_step(update_counter.cur_checkpoint)
                if self.model_keys is None:
                    self.model_keys = [
                        self._model_to_key(model, model_kwargs)
                        for model, model_kwargs in zip(models, self.models_kwargs)
                    ]
                for i in range(len(self.model_keys)):
                    self.models[f"{extractor}.{self.model_keys[i]}"] = models[i]

        # train from features
        for extractor, features in zip(self.extractors, all_features):
            for model_key in self.model_keys:
                probe_model = self.models[f"{extractor}.{model_key}"]
                probe_model.before_accumulation_step()
                probe_model.train()
                with torch.enable_grad():
                    with trainer.autocast_context:
                        preds = probe_model(features)
                        loss = self.loss_function(preds=preds, images=x) / accumulation_steps
                        trainer.grad_scaler.scale(loss).backward()
                model_name = f"{extractor}.{model_key}"
                self.tracked_losses[model_name].append(loss.item())
                # this doesn't work as the update_counter wasn't increased yet, so after an epoch the logger fails
                # if len(self.tracked_losses[model_name]) % 50 == 0:
                #     mean_loss = all_reduce_mean_grad(np.mean(self.tracked_losses[model_name][-50:]))
                #     self.writer.add_scalar(
                #         f"{self.base_key}/{model_name}/U50",
                #         mean_loss,
                #         update_counter=update_counter,
                #     )

    def _forward(self, batch, model, trainer):
        batch_without_ctx, _ = batch  # remove ctx
        images = self.denormalize(ModeWrapper.get_item(mode=trainer.dataset_mode, item="x", batch=batch_without_ctx))
        images = images.to(model.device, non_blocking=True)

        losses = {}
        with trainer.autocast_context:
            trainer.forward(model=model, batch=batch, train_dataset=self.dataset)
            for extractor in self.extractors:
                features = extractor.extract()
                for model_key in self.model_keys:
                    model = self.models[f"{extractor}.{model_key}"]
                    preds = model.predict(features)
                    for pred_name, pred in preds.items():
                        loss = self.loss_function_noreduce(preds=pred, images=images)
                        losses[(f"{extractor}.{model_key}", pred_name)] = loss.cpu()
        return losses

    # noinspection PyMethodOverriding
    def _log(self, update_counter, interval_type, model, trainer, **_):
        # log train loss
        if len(self.tracked_losses) > 0:
            for model_name, loss in self.tracked_losses.items():
                mean_loss = all_reduce_mean_grad(np.mean(loss))
                self.logger.info(f"{self.base_key}/{model_name}: {mean_loss:.4f}")
                self.writer.add_scalar(
                    f"{self.base_key}/{model_name}/{self.to_short_interval_string()}",
                    mean_loss,
                    update_counter=update_counter,
                )
            self.tracked_losses.clear()

        # calculate test metrics
        for probe_model in self.models.values():
            probe_model.eval()
        for extractor in self.extractors:
            extractor.enable_hooks()
        test_losses = self.iterate_over_dataset_collated(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            update_counter=update_counter,
        )
        for extractor in self.extractors:
            extractor.disable_hooks()
        # push to GPU for accuracy calculation
        test_losses = {k: v.to(model.device, non_blocking=True) for k, v in test_losses.items()}
        # log
        for (extractor_model_name, prediction_name), test_loss in test_losses.items():
            mean_loss = test_loss.mean().item()
            key = f"{self.base_key}/{extractor_model_name}/{self.dataset_key}/{prediction_name}"
            self.logger.info(f"{key}: {mean_loss:.4f}")
            self.writer.add_scalar(key, mean_loss, update_counter=update_counter)

    def _after_training(self, **_):
        for extractor in self.extractors:
            for model_key in self.model_keys:
                model = self.models[f"{extractor}.{model_key}"]
                model.name = f"{extractor}.{model_key}"
                self.checkpoint_writer.save(
                    model=model,
                    checkpoint="last",
                    save_weights=True,
                    save_optim=False,
                )
