from collections import defaultdict
from functools import partial

import torch

from distributed.gather import all_gather_nograd_clipped
from .base.multi_dataset_logger import MultiDatasetLogger
from utils.factory import create_collection
from models.extractors import extractor_from_kwargs
from datasets.image_net import ImageNet
from initializers.base.checkpoint_initializer import CheckpointInitializer
from utils.subset_identifier import get_subset_identifier
from distributed.config import is_rank0
from kappadata import ModeWrapper


class OfflineFeatureLogger(MultiDatasetLogger):
    def __init__(self, dataset_key, extractors, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        # create output folder
        self.out_folder = self.stage_path_provider.stage_output_path / "features"
        self.out_folder.mkdir(exist_ok=True)

    def _before_training(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, model, trainer, train_dataset):
        features = {}
        with trainer.autocast_context:
            trainer.forward(model=model, batch=batch, train_dataset=train_dataset)
            for extractor in self.extractors:
                features[str(extractor)] = extractor.extract().cpu()
        batch, _ = batch  # remove ctx
        classes = ModeWrapper.get_item(mode=trainer.dataset_mode, item="class", batch=batch)
        return features, classes.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, train_dataset, **_):
        #assert trainer.precision == torch.float32
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # extract
        features, labels = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, train_dataset=train_dataset),
            dataset_key=self.dataset_key,
            dataset_mode=trainer.dataset_mode,
            batch_size=trainer.effective_batch_size,
            update_counter=update_counter,
            persistent_workers=False,
        )

        # log
        for feature_name, feature in features.items():
            fname = f"{self.dataset_key}-{feature_name}-{update_counter.cur_checkpoint}-features.th"
            if is_rank0():
                torch.save(feature, self.out_folder / fname)
            self.logger.info(f"wrote features to {fname}")
        torch.save(labels, self.out_folder / f"{self.dataset_key}-labels.th")

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
