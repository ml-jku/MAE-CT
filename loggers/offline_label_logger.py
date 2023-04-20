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


class OfflineLabelLogger(MultiDatasetLogger):
    def __init__(self, dataset_key, stage_id=None, force=False, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.force = force
        self.stage_id = stage_id
        # create output folder
        self.out_folder = self.stage_path_provider.output_path / "labels"
        self.out_folder.mkdir(exist_ok=True)

    @staticmethod
    def _forward(batch):
        return batch[0]

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, **_):
        if self.stage_id is None:
            # extract stage_id from initializer
            initializers = []
            for submodel in model.submodels.values():
                if isinstance(submodel.initializer, CheckpointInitializer):
                    initializers.append(submodel.initializer)
            initializer_stage_ids = [initializer.stage_id for initializer in initializers]
            if len(initializer_stage_ids) > 1:
                assert all(initializer_stage_ids[0] == isd for isd in initializer_stage_ids[1:])
            stage_id = initializer_stage_ids[0]
        else:
            stage_id = self.stage_id
        out_folder = self.out_folder / stage_id
        out_folder.mkdir(exist_ok=True)

        # compose fname
        dataset = self.data_container.get_dataset(self.dataset_key)
        dataset_identifier = str(dataset.root_dataset)
        subset_identifier = get_subset_identifier(dataset)
        key = f"{dataset_identifier}{subset_identifier}-labels.th"
        fname = out_folder / key
        if fname.exists():
            if not self.force:
                self.logger.info(f"labels '{fname}' already exists -> skip")
                return
            self.logger.info(f"labels '{fname}' already exists -> overwrite")

        # extract
        labels = self.iterate_over_dataset(
            forward_fn=self._forward,
            dataset_key=self.dataset_key,
            dataset_mode="class",
            batch_size=trainer.effective_batch_size,
            update_counter=update_counter,
        )

        # log
        if is_rank0():
            torch.save(labels, fname)
        self.logger.info(f"wrote labels to {fname}")