import torch
from kappadata import ModeWrapper, KDMixWrapper
from kappadata.utils.class_counts import get_class_counts

from loggers.base.summary_logger import SummaryLogger
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR


class DatasetStatsLogger(SummaryLogger):
    @property
    def allows_no_interval_types(self):
        return True

    def _before_training(self, update_counter, **_):
        for dataset_key, dataset in self.data_container.datasets.items():
            self._log_size(dataset_key, dataset)
            self._log_class_label_statistics(dataset_key, dataset)

    def _log_size(self, dataset_key, dataset):
        self.summary_provider[f"ds_stats/{dataset_key}/len"] = len(dataset)
        self.logger.info(f"{dataset_key}: {len(dataset)} samples")

    def _log_class_label_statistics(self, dataset_key, dataset):
        # check if dataset has class labels
        if not hasattr(dataset, "getitem_class"):
            return

        # skip too large/too small
        if len(dataset) > 1e5:
            self.logger.info(f"skipping dataset statistics for {dataset_key} (too big len(ds)={len(dataset)})")
            return
        if len(dataset) == 0:
            self.logger.info(f"skipping dataset statistics for {dataset_key} (len(ds)==0)")
            return
        if isinstance(dataset.root_dataset, (ImageNetA, ImageNetR)):
            self.logger.info(f"skipping dataset statistics for {dataset_key} (ImageNet-A/R not supported yet)")
            return

        # cls_loader = DataLoader(ModeWrapper(dataset, mode="class"), batch_size=4096, num_workers=get_fair_cpu_count())
        # classes = []
        # for cls in cls_loader:
        #     classes.append(cls.clone())
        # classes = torch.concat(classes)

        if dataset.has_wrapper_type(KDMixWrapper):
            return

        classes = ModeWrapper(dataset, mode="class")[:]
        if dataset.is_multiclass:
            self._log_class_counts(
                dataset_key=dataset_key,
                dataset=dataset,
                classes=classes,
                n_classes=dataset.n_classes,
                is_multiclass=True,
            )
        elif (torch.is_tensor(classes[0]) and classes[0].dtype == torch.long) or isinstance(classes[0], int):
            self._log_class_counts(
                dataset_key=dataset_key,
                dataset=dataset,
                classes=classes,
                n_classes=dataset.n_classes,
                is_multiclass=False,
            )

    def _log_class_counts(self, dataset_key, dataset, classes, n_classes, is_multiclass):
        if is_multiclass:
            counts = torch.stack(classes).sum(dim=0)
        else:
            counts = get_class_counts(classes, n_classes)
        nonzero_counts = counts[counts > 0]
        self.logger.info(f"{dataset_key} has {n_classes} classes ({len(nonzero_counts)} classes with samples)")
        if len(nonzero_counts) <= 10:
            class_names = dataset.class_names if hasattr(dataset, "class_names") else None
            for i in range(len(nonzero_counts)):
                class_name = i if class_names is None else class_names[i]
                # with wandb this creates an implicit summary which might be confusing
                # self.writer.add_scalar(f"ds_stats/{dataset_key}/classes/", counts[i])
                self.logger.info(f"{nonzero_counts[i]} {nonzero_counts[i] / len(classes) * 100:.2f}% {class_name}")
        self.logger.info(f"each class has at least {counts.min()} samples")
        self.logger.info(f"each class has at most {counts.max()} samples")
        #self.logger.info(f"each class has on average {counts.float().mean()} samples")
