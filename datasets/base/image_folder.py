import os
import shutil

import yaml
from kappadata import SharedDictDataset
from kappadata.copying.image_folder import copy_imagefolder_from_global_to_local
from kappadata.loading.image_folder import raw_image_folder_sample_to_pil_sample, raw_image_loader

from distributed.config import barrier, is_data_rank0
from utils.num_worker_heuristic import get_fair_cpu_count
from utils.param_checking import to_path
from .tv_image_folder import TVImageFolder
from .xtransform_dataset_base import XTransformDatasetBase
from pathlib import Path


class ImageFolder(XTransformDatasetBase):
    def __init__(self, global_root=None, local_root=None, caching_mode=None, **kwargs):
        super().__init__(**kwargs)
        # automatically populate global_root/local_root/caching_mode if they are not defined explicitly
        if global_root is None:
            global_root = self.dataset_config_provider.get_global_dataset_path(self.get_dataset_identifier())
        else:
            global_root = to_path(global_root)
        if local_root is None:
            if self.dataset_config_provider is not None:
                source_mode = self.dataset_config_provider.get_data_source_mode(self.get_dataset_identifier())
                # use local by default
                if source_mode in [None, "local"]:
                    local_root = self.dataset_config_provider.get_local_dataset_path()
        else:
            local_root = to_path(local_root)
        if caching_mode is None and self.dataset_config_provider is not None:
            caching_mode = self.dataset_config_provider.get_data_caching_mode(self.get_dataset_identifier())
        # get relative path (e.g. train)
        relative_path = self.get_relative_path()
        if local_root is None:
            # load data from global_root
            assert global_root is not None and global_root.exists(), f"invalid global_root '{global_root}'"
            source_root = global_root / relative_path
            assert source_root.exists(), f"invalid source_root (global) '{source_root}'"
            self.logger.info(f"data_source (global): '{source_root}'")
        else:
            # load data from local_root
            source_root = local_root / self.get_dataset_identifier() / relative_path
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root / relative_path}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                copy_imagefolder_from_global_to_local(
                    global_path=global_root,
                    local_path=local_root / self.get_dataset_identifier(),
                    relative_path=relative_path,
                    # on karolina 5 was already too much for a single GPU
                    # "A worker process managed by the executor was unexpectedly terminated.
                    # This could be caused by a segmentation fault while calling the function or by an
                    # excessive memory usage causing the Operating System to kill the worker."
                    num_workers=min(10, get_fair_cpu_count()),
                    log_fn=self.logger.info,
                )
                # check folder structure
                folders = [None for f in os.listdir(source_root) if (source_root / f).is_dir()]
                self.logger.info(f"source_root '{source_root}' contains {len(folders)} folders")
            barrier()

        # initialize caching strategy
        if caching_mode is None:
            self.dataset = TVImageFolder(source_root)
        elif caching_mode == "shared_dict":
            self.logger.info(f"data_caching_mode: shared_dict")
            ds = TVImageFolder(source_root, loader=raw_image_loader)
            self.dataset = SharedDictDataset(dataset=ds, transform=raw_image_folder_sample_to_pil_sample)
        else:
            raise NotImplementedError

    def get_dataset_identifier(self):
        """ returns an identifier for the dataset (used for retrieving paths from dataset_config_provider) """
        raise NotImplementedError

    def get_relative_path(self):
        """
        return the relative path to the dataset root
        - e.g. /train (ImageNet)
        - e.g. /bottle (MVTec)
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        x = self.x_transform(x, ctx=ctx)
        return x

    def getitem_class(self, idx, ctx=None):
        return self.dataset.targets[idx]

    # noinspection PyUnusedLocal
    def getitem_fname(self, idx, ctx=None):
        return str(Path(self.dataset.samples[idx][0]).relative_to(self.dataset.root))

    @property
    def n_classes(self):
        return len(self.dataset.classes)
