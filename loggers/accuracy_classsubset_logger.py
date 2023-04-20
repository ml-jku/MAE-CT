from collections import defaultdict
from functools import partial

import torch
from torchmetrics.functional.classification import multiclass_accuracy

from .base.multi_dataset_logger import MultiDatasetLogger
from distributed.gather import all_gather_nograd_clipped
from .base.dataset_logger import DatasetLogger


class AccuracyClasssubsetLogger(MultiDatasetLogger):
    def __init__(self, dataset_key, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.class_subset_indices = None
        self.n_classes = None

    def _before_training(self, **kwargs):
        dataset = self.data_container.get_dataset(self.dataset_key)
        self.class_subset_indices = dataset.class_subset_indices
        self.n_classes = dataset.n_classes

    def _forward(self, batch, model, trainer):
        (x, cls), _ = batch
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model.predict(x)
        # only use logits for the actual available classes
        predictions = {
            name: prediction[:, self.class_subset_indices].cpu()
            for name, prediction in predictions.items()
        }
        return predictions, cls.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, logger_info_dict, **_):
        # extract
        predictions, classes = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            dataset_key=self.dataset_key,
            dataset_mode="x class",
            batch_size=trainer.effective_batch_size,
            update_counter=update_counter,
        )

        # push to GPU for accuracy calculation
        predictions = {k: v.to(model.device, non_blocking=True) for k, v in predictions.items()}
        classes = classes.to(model.device, non_blocking=True)

        # log
        for prediction_name, prediction in predictions.items():
            for topk in [1]:
                acc = multiclass_accuracy(
                    preds=prediction,
                    target=classes,
                    top_k=topk,
                    num_classes=self.n_classes,
                    average="micro",
                ).item()
                key = f"accuracy{topk}/{self.dataset_key}/{prediction_name}"
                self.logger.info(f"{key}: {acc:.4f}")
                logger_info_dict[key] = acc
                self.writer.add_scalar(key, acc, update_counter=update_counter)

