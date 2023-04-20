from collections import defaultdict
from functools import partial

import torch
from torchmetrics.functional.classification import multiclass_accuracy

from distributed.gather import all_gather_nograd_clipped
from .base.dataset_logger import DatasetLogger


class AccuracyLogger(DatasetLogger):
    def __init__(self, predict_kwargs=None, accuracies_per_class=False, **kwargs):
        super().__init__(**kwargs)
        self.predict_kwargs = predict_kwargs or {}
        self.accuracies_per_class = accuracies_per_class

    def _before_training_impl(self, **kwargs):
        if self.dataset.n_classes <= 10:
            self.top_k = [1]
        else:
            # calculating an additional accuracy for IN1K is not that cheap (ViT-B probe: 20% of update time)
            # self.top_k = [1, 5]
            self.top_k = [1]

    def _forward_accuracy(self, batch, model, trainer):
        x, cls = batch
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model.predict(x, **self.predict_kwargs)
        predictions = {name: prediction.cpu() for name, prediction in predictions.items()}
        return predictions, cls.clone()

    def get_dataset_mode(self, *_, **__):
        return "x class"

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, logger_info_dict, **_):
        forward_results = self.iterate_over_dataset(
            forward_fn=partial(self._forward_accuracy, model=model, trainer=trainer),
            update_counter=update_counter,
        )
        prediction_batches = defaultdict(list)
        class_batches = []
        for forward_result in forward_results:
            prediction_batch, label_batch = forward_result
            for key, value in prediction_batch.items():
                prediction_batches[key].append(value)
            class_batches.append(label_batch)
        predictions = {key: torch.concat(value) for key, value in prediction_batches.items()}
        classes = torch.concat(class_batches)

        # gather
        predictions = {
            k: all_gather_nograd_clipped(v, len(self.dataset))
            for k, v in predictions.items()
        }
        classes = all_gather_nograd_clipped(classes, len(self.dataset))

        # log
        classes = classes.to(model.device, non_blocking=True)
        for prediction_name, prediction in predictions.items():
            prediction = prediction.to(model.device, non_blocking=True)
            for topk in self.top_k:
                acc = multiclass_accuracy(
                    preds=prediction,
                    target=classes,
                    top_k=topk,
                    num_classes=self.data_container.train.n_classes,
                    average="micro",
                ).item()
                key = f"accuracy{topk}/{self.dataset_key}/{prediction_name}"
                self.logger.info(f"{key}: {acc:.4f}")
                logger_info_dict[key] = acc
                self.writer.add_scalar(key, acc, update_counter=update_counter)

        # accuracies per class
        if self.accuracies_per_class:
            class_names = self.dataset.class_names or [f"class{i}" for i in range(self.dataset.n_classes)]
            for prediction_name, prediction in predictions.items():
                prediction = prediction.to(model.device)
                for topk in self.top_k:
                    accs_per_class = multiclass_accuracy(
                        preds=prediction,
                        target=classes,
                        top_k=topk,
                        num_classes=self.data_container.train.n_classes,
                        average="none",
                    )
                    for i in range(self.dataset.n_classes):
                        key = (
                            f"accuracy{topk}_per_class/{self.dataset_key}/"
                            f"{class_names[i]}/{prediction_name}"
                        )
                        self.logger.info(f"{key}: {accs_per_class[i]:.4f}")
                        self.writer.add_scalar(key, accs_per_class[i], update_counter=update_counter)
