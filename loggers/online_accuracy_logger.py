from collections import defaultdict

import numpy as np
import torch
from torchmetrics.functional.classification import multiclass_accuracy

from distributed.gather import all_reduce_mean_grad
from loggers.base.logger_base import LoggerBase


class OnlineAccuracyLogger(LoggerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracked_accs = defaultdict(lambda: defaultdict(list))
        self.top_k = None

    @property
    def allows_multiple_interval_types(self):
        return False

    def _before_training(self, model, **kwargs):
        assert len(model.output_shape) == 1
        if model.output_shape[0] <= 10:
            self.top_k = [1]
        else:
            # calculating an additional accuracy for IN1K is not that cheap (ViT-B probe: 20% of update time)
            self.top_k = [1]
            # self.top_k = [1, 5]

    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        classes = update_outputs["classes"]
        # convert back to long (e.g. when label smoothing is used)
        if classes.dtype != torch.long:
            classes = classes.argmax(dim=1)

        for name, prediction in update_outputs["predictions"].items():
            for topk in self.top_k:
                acc = multiclass_accuracy(
                    preds=prediction,
                    target=classes,
                    top_k=topk,
                    num_classes=self.data_container.train.n_classes,
                    average="micro",
                ).item()
                self.tracked_accs[name][topk].append(acc)

    def _log(self, update_counter, interval_type, **_):
        for name, tracked_prediction in self.tracked_accs.items():
            for topk, tracked_acc in tracked_prediction.items():
                mean_acc = np.mean(tracked_acc)
                mean_acc = all_reduce_mean_grad(mean_acc)
                self.logger.info(f"accuracy{topk}/online/{name}: {mean_acc:.4f}")
                self.writer.add_scalar(
                    f"accuracy{topk}/online/{name}/{self.to_short_interval_string()}",
                    mean_acc,
                    update_counter=update_counter,
                )
        self.tracked_accs.clear()
