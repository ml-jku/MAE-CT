from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader
from losses.bce_loss import bce_loss

import numpy as np
import torch
from kappadata import ModeWrapper
from torch.nn.functional import cross_entropy
from torchmetrics.functional.classification import multiclass_accuracy, binary_accuracy, binary_auroc

from distributed.gather import all_reduce_mean_grad, all_gather_nograd
from loggers.base.dataset_logger import DatasetLogger
from models import model_from_kwargs
from models.extractors import extractor_from_kwargs
from utils.factory import create_collection
from utils.formatting_util import float_to_scientific_notation
from utils.select_with_path import select_with_path
from metrics.functional.auprc import auprc


class OnlineProbeLogger(DatasetLogger):
    def __init__(self, extractors, models, **kwargs):
        super().__init__(**kwargs)
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        self.tracked_accs = defaultdict(list)
        self.tracked_preds = defaultdict(list)
        self.tracked_classes = []
        self.models_kwargs = models
        self.model_keys = None
        self.models = None

    def get_dataset_mode(self, trainer):
        return trainer.dataset_mode

    def return_ctx(self):
        return True

    @property
    def allows_multiple_interval_types(self):
        return False

    def _before_training_impl(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)

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
        for classes_key in ["cls", "classes"]:
            if classes_key in update_outputs:
                classes = update_outputs[classes_key]
                break
        else:
            raise NotImplementedError
        all_features = [extractor.extract().detach() for extractor in self.extractors]

        if self.models is None:
            self.models = {}
            for extractor, features in zip(self.extractors, all_features):
                models = create_collection(
                    self.models_kwargs,
                    model_from_kwargs,
                    input_shape=features.shape[1:],
                    output_shape=(train_dataset.n_classes,),
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
                        if train_dataset.n_classes > 2:
                            loss = cross_entropy(preds, classes)
                        else:
                            loss = bce_loss(preds, classes)
                        loss = loss / accumulation_steps
                        trainer.grad_scaler.scale(loss).backward()
                if train_dataset.n_classes > 2:
                    train_acc = multiclass_accuracy(
                        preds=preds,
                        target=classes,
                        num_classes=train_dataset.n_classes,
                        average="micro",
                    ).item()
                    self.tracked_accs[f"{extractor}.{model_key}"].append(train_acc)
                else:
                    preds = preds.squeeze(1)
                    # train_acc = binary_accuracy(preds=preds, target=classes)
                    # self.tracked_accs[f"{extractor}.{model_key}"].append(train_acc.item())
                    # auroc can't be calculated on-the-fly
                    self.tracked_preds[f"{extractor}.{model_key}"].append(preds.detach().cpu())
                    self.tracked_classes.append(classes.cpu())

    def _forward(self, batch, model, trainer):
        predictions = {}
        with trainer.autocast_context:
            trainer.forward(model=model, batch=batch, train_dataset=self.dataset)
            for extractor in self.extractors:
                features = extractor.extract()
                for model_key in self.model_keys:
                    model = self.models[f"{extractor}.{model_key}"]
                    preds = model.predict(features)
                    for pred_name, pred in preds.items():
                        predictions[(f"{extractor}.{model_key}", pred_name)] = pred.cpu()
        batch, _ = batch  # remove ctx
        classes = ModeWrapper.get_item(mode=trainer.dataset_mode, item="class", batch=batch)
        return predictions, classes.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, interval_type, model, trainer, **_):
        # log train accuracies
        if len(self.tracked_accs) > 0:
            for model_name, accs in self.tracked_accs.items():
                mean_acc = all_reduce_mean_grad(np.mean(accs))
                self.logger.info(f"accuracy1/onlineprobe/{model_name}: {mean_acc:.4f}")
                self.writer.add_scalar(
                    f"accuracy1/onlineprobe/{model_name}/{self.to_short_interval_string()}",
                    mean_acc,
                    update_counter=update_counter,
                )
            self.tracked_accs.clear()
        # log train auroc
        if len(self.tracked_preds) > 0:
            classes = all_gather_nograd(torch.concat(self.tracked_classes))
            # subsample training preds/classes (with 1M this results in a OOM error otherwise when calculating metrics)
            perm = torch.randperm(len(classes))[:len(self.dataset)]
            classes = classes[perm].to(model.device)
            for model_name, preds in self.tracked_preds.items():
                # subsample preds
                preds = all_gather_nograd(torch.concat(preds))[perm].to(model.device)
                if (classes == 0).sum() == 0 or (classes == 1).sum() == 0:
                    train_auroc = 1.
                    train_auprc = 1.
                    self.logger.warning(
                        f"only samples for one class present ({(classes == 0).sum()} - {(classes == 1).sum()}) "
                        f"-> perfect train_auroc/train_auprc"
                    )
                else:
                    train_auroc = binary_auroc(preds=preds, target=classes)
                    train_auprc = auprc(preds=preds, target=classes)
                self.logger.info(f"auroc/onlineprobe/{model_name}: {train_auroc:.4f}")
                self.logger.info(f"auprc/onlineprobe/{model_name}: {train_auprc:.4f}")
                self.writer.add_scalar(
                    f"auroc/onlineprobe/{model_name}/{self.to_short_interval_string()}",
                    train_auroc,
                    update_counter=update_counter,
                )
                self.writer.add_scalar(
                    f"auprc/onlineprobe/{model_name}/{self.to_short_interval_string()}",
                    train_auprc,
                    update_counter=update_counter,
                )
            self.tracked_classes.clear()
            self.tracked_preds.clear()

        # calculate test metrics
        for probe_model in self.models.values():
            probe_model.eval()
        for extractor in self.extractors:
            extractor.enable_hooks()
        predictions, classes = self.iterate_over_dataset_collated(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            update_counter=update_counter,
        )
        for extractor in self.extractors:
            extractor.disable_hooks()
        # push to GPU for accuracy calculation
        predictions = {k: v.to(model.device, non_blocking=True) for k, v in predictions.items()}
        classes = classes.to(model.device, non_blocking=True)
        # log
        for (extractor_model_name, prediction_name), prediction in predictions.items():
            if self.data_container.train.n_classes > 2:
                test_acc = multiclass_accuracy(
                    preds=prediction,
                    target=classes,
                    num_classes=self.data_container.train.n_classes,
                    average="micro",
                ).item()
                key = f"accuracy1/onlineprobe/{extractor_model_name}/{self.dataset_key}/{prediction_name}"
                self.logger.info(f"{key}: {test_acc:.4f}")
                self.writer.add_scalar(key, test_acc, update_counter=update_counter)
            else:
                prediction = prediction.squeeze(1)
                #test_acc = binary_accuracy(preds=prediction, target=classes)
                if (classes == 0).sum() == 0 or (classes == 1).sum() == 0:
                    test_auroc = 1.
                    test_auprc = 1.
                    self.logger.warning(
                        f"only samples for one class present ({(classes == 0).sum()} - {(classes == 1).sum()}) "
                        f"-> perfect test_auroc/test_auprc"
                    )
                else:
                    test_auroc = binary_auroc(preds=prediction, target=classes)
                    test_auprc = auprc(preds=prediction, target=classes)
                key = f"onlineprobe/{extractor_model_name}/{self.dataset_key}/{prediction_name}"
                #self.logger.info(f"accuracy/{key}: {test_acc:.4f}")
                self.logger.info(f"auroc/{key}: {test_auroc:.4f}")
                self.logger.info(f"auprc/{key}: {test_auprc:.4f}")
                #self.writer.add_scalar(f"accuracy/{key}", test_acc, update_counter=update_counter)
                self.writer.add_scalar(f"auroc/{key}", test_auroc, update_counter=update_counter)
                self.writer.add_scalar(f"auprc/{key}", test_auprc, update_counter=update_counter)
