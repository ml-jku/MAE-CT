from functools import partial

from kappadata import ModeWrapper
from metrics.functional.auprc import auprc
from torchmetrics.functional.classification import binary_auroc

from loggers.base.multi_dataset_logger import MultiDatasetLogger
from metrics.functional.knn import knn_metrics
from models.extractors import extractor_from_kwargs
from utils.factory import create_collection
from utils.formatting_util import dict_to_string
from utils.object_from_kwargs import objects_from_kwargs


class KnnMetricsLogger(MultiDatasetLogger):
    def __init__(
            self,
            train_dataset_key,
            test_dataset_key,
            extractors,
            knns=None,
            forward_kwargs=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataset_key = train_dataset_key
        self.test_dataset_key = test_dataset_key
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.knns = knns or [1]

    def _before_training(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, model, trainer, train_dataset):
        features = {}
        with trainer.autocast_context:
            trainer.forward(model=model, batch=batch, train_dataset=train_dataset, **self.forward_kwargs)
            for extractor in self.extractors:
                features[str(extractor)] = extractor.extract().cpu()
        batch, _ = batch  # remove ctx
        classes = ModeWrapper.get_item(mode=trainer.dataset_mode, item="class", batch=batch)
        return features, classes.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, interval_type, model, trainer, train_dataset, logger_info_dict, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # source_dataset foward (this is the "queue" from the online nn_accuracy)
        train_features, train_y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, train_dataset=train_dataset),
            dataset_key=self.train_dataset_key,
            dataset_mode=trainer.dataset_mode,
            batch_size=trainer.effective_batch_size,
            update_counter=update_counter,
            persistent_workers=False,
        )
        # target_dataset forward (this is the "test" dataset from the online nn_accuracy)
        test_features, test_y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, train_dataset=train_dataset),
            dataset_key=self.test_dataset_key,
            dataset_mode=trainer.dataset_mode,
            batch_size=trainer.effective_batch_size,
            update_counter=update_counter,
            persistent_workers=False,
        )

        # calculate/log metrics
        train_y = train_y.to(model.device)
        test_y = test_y.to(model.device)
        # take only 1st view of the source features (just like NNCLR does it)
        train_features = {k: v[:len(train_y)] for k, v in train_features.items()}
        for feature_key in train_features.keys():
            train_x = train_features[feature_key].to(model.device)
            test_x = test_features[feature_key].to(model.device)
            assert len(test_x) == len(test_y), "expecting single view input"

            # calculate
            for batch_normalize in [True]:  # [False, True]:
                accuracies, purities, scores = knn_metrics(
                    train_x=train_x,
                    test_x=test_x,
                    train_y=train_y,
                    test_y=test_y,
                    knns=self.knns,
                    batch_size=trainer.effective_batch_size,
                    batch_normalize=batch_normalize,
                )

                # log (per view)
                forward_kwargs_str = f"/{dict_to_string(self.forward_kwargs)}" if len(self.forward_kwargs) > 0 else ""
                for knn in purities.keys():
                    feature_key_bn = f"{feature_key}-batchnorm" if batch_normalize else feature_key
                    key = (
                        f"knn{knn}/{feature_key_bn}/{self.train_dataset_key}-{self.test_dataset_key}"
                        f"{forward_kwargs_str}"
                    )
                    purity = purities[knn]
                    # LEGACY this should be called knn_purity but it is kept as nn_purity because old runs used it
                    self.logger.info(f"nn_purity/{key}: {purity:.4f}")
                    self.writer.add_scalar(f"nn_purity/{key}", purity, update_counter=update_counter)
                    logger_info_dict[f"nn_purity/{key}"] = purity
                    if scores is None:
                        accuracy = accuracies[knn]
                        self.logger.info(f"knn_accuracy/{key}: {accuracy:.4f}")
                        self.writer.add_scalar(f"knn_accuracy/{key}", accuracy, update_counter=update_counter)
                        logger_info_dict[f"knn_accuracy/{key}"] = accuracy
                    else:
                        test_auroc = binary_auroc(preds=scores[knn], target=test_y)
                        test_auprc = auprc(preds=scores[knn], target=test_y)
                        self.logger.info(f"knn_auroc/{key}: {test_auroc:.4f}")
                        self.logger.info(f"knn_auprc/{key}: {test_auprc:.4f}")
                        self.writer.add_scalar(f"knn_auroc/{key}", test_auroc, update_counter=update_counter)
                        self.writer.add_scalar(f"knn_auprc/{key}", test_auprc, update_counter=update_counter)
                        logger_info_dict[f"knn_auroc/{key}"] = test_auroc
                        logger_info_dict[f"knn_auprc/{key}"] = test_auprc

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
