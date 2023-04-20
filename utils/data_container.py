import logging
from functools import partial

import numpy as np
import kappadata as kd
from kappadata import ModeWrapper, KDComposeCollator, SubsetWrapper, ShuffleWrapper
from pytorch_concurrent_dataloader import DataLoader as ConcurrentDataloader
from torch.utils.data import DistributedSampler, DataLoader, RandomSampler, SequentialSampler

from distributed.config import is_distributed, get_world_size
from providers.config_providers.noop_config_provider import NoopConfigProvider
from utils.infinite_batch_sampler import InfiniteBatchSampler
from utils.num_worker_heuristic import get_num_workers, get_total_cpu_count, get_num_fetch_workers
from datasets.dummy_dataset import DummyDataset


class DataContainer:
    def __init__(
            self,
            num_workers=None,
            config_provider=None,
            generator=None,
            **datasets,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.num_workers = num_workers
        self.config_provider = config_provider or NoopConfigProvider()
        self.generator = generator

        self.datasets = datasets
        self.persistent_loaders = {}
        self.added_to_config_provider = False
        # run_type can be adjusted by trainers
        self.run_type = "train"

    @property
    def train(self):
        return self.get_dataset("train")

    def get_dataset(self, key, mode=None, max_size=None, return_ctx=False, shuffle_seed=None):
        dataset = self.datasets[key]
        if max_size is not None:
            dataset = SubsetWrapper(dataset, end_index=max_size)
        if shuffle_seed is not None:
            dataset = ShuffleWrapper(dataset=dataset, seed=shuffle_seed)
        if mode is not None:
            dataset = ModeWrapper(dataset=dataset, mode=mode, return_ctx=return_ctx)
        return dataset

    # TODO better naming for is_train_dataset
    def dataloader_from_key(
            self,
            dataset_key,
            mode,
            batch_size,
            is_train_dataset,
            shuffle=None,
            drop_last=None,
            return_ctx=False,
            max_size=None,
            num_workers=None,
            shuffle_seed=None,
            end_checkpoint=None,
            persistent_workers=None,
    ):
        # get dataset
        dataset = self.get_dataset(
            key=dataset_key,
            mode=mode,
            max_size=max_size,
            return_ctx=return_ctx,
            shuffle_seed=shuffle_seed,
        )
        if max_size is not None:
            batch_size = min(batch_size, max_size)

        # create persistent loader
        if dataset_key == "train" and is_train_dataset:
            shuffle = True if shuffle is None else shuffle
            drop_last = True if drop_last is None else drop_last
            loader = self.dataloader_from_dataset(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                is_train_dataset=True,
                num_workers=num_workers,
                end_checkpoint=end_checkpoint,
                persistent_workers=persistent_workers,
            )
        else:
            assert end_checkpoint is None
            loader = self.dataloader_from_dataset(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle or False,
                drop_last=drop_last or False,
                is_train_dataset=False,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            )
        if isinstance(loader, DataLoader):
            dataloader_class = "pytorch"
            num_fetch_workers_str = ""
        elif isinstance(loader, ConcurrentDataloader):
            dataloader_class = "concurrent"
            num_fetch_workers_str = f" num_fetch_workers={loader.num_fetch_workers}"
        else:
            raise NotImplementedError
        self.logger.info(
            f"created '{dataset_key}' dataloader (type={dataloader_class} batch_size={batch_size} "
            f"num_workers={loader.num_workers}{num_fetch_workers_str} pin_memory={loader.pin_memory} "
            f"dataset_length={len(dataset)} persistent_workers={loader.persistent_workers} "
            f"total_cpu_count={get_total_cpu_count()})"
        )
        # add to wandb config
        if not self.added_to_config_provider:
            self.config_provider.update({
                f"dataloader/{dataset_key}/num_workers": loader.num_workers,
                f"dataloader/{dataset_key}/pin_memory": loader.pin_memory,
                f"dataloader/{dataset_key}/persistent_workers": loader.persistent_workers,
                f"dataloader/{dataset_key}/dataloader_class": dataloader_class,
            })
            self.added_to_config_provider = True
        return loader

    def dataloader_from_dataset(
            self,
            dataset,
            batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=None,
            is_train_dataset=False,
            end_checkpoint=None,
            persistent_workers=None,
    ):
        # create collator
        if dataset.collators is not None and len(dataset.collators) > 0:
            collator = KDComposeCollator(
                collators=dataset.collators,
                dataset_mode=dataset.mode,
                return_ctx=dataset.return_ctx,
            )
        else:
            collator = None

        # check dataset size with batch_size
        if drop_last:
            if len(dataset) < batch_size:
                self.logger.warning(
                    f"dataset is too small to drop_last ({len(bs)}<{batch_size}) "
                    f"-> using batch_size=len(dataset) and drop_last=False"
                )
                batch_size = len(dataset)
                drop_last = False
        elif len(dataset) < batch_size:
            self.logger.info(
                f"dataset smaller than batch_size ({len(dataset)}<{batch_size}) "
                f"-> using batch_size=len(dataset)"
            )
            batch_size = min(len(dataset), batch_size)
            # distributed edge cases not implemented yet
            if is_distributed():
                assert batch_size % get_world_size() == 0

        # distributed sampler if necessary
        if is_distributed():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        else:
            sampler = None

        # if train is a DummyDataset -> eval run -> use all workers for test dataset
        if isinstance(self.train.root_dataset, DummyDataset):
            n_datasets = 1
        else:
            n_datasets = len(self.datasets)

        num_worker_heuristic = get_num_workers(
            dataset=dataset,
            batch_size=batch_size,
            n_datasets=n_datasets,
            is_train_dataset=is_train_dataset,
            run_type=self.run_type,
        )
        num_workers = num_workers or self.num_workers or num_worker_heuristic
        num_fetch_workers = get_num_fetch_workers(dataset=dataset)
        pin_memory = True
        kwargs = {}
        if num_workers > 0:
            if num_fetch_workers > 0:
                # ConcurrentDataloader doesn't support persistent workers
                persistent_workers = False
                # TODO i think ConcurrentDataloader has an issue with pin_memory
                pin_memory = False
            else:
                if persistent_workers is None:
                    persistent_workers = True

            # if num_workers > n_batches -> avoid unnecessary processes
            # only used for test datasets (for small datasets batches from the next epochs can already be prefetched)
            if not is_train_dataset:
                n_batches = int(np.ceil(len(dataset) / batch_size))
                if num_workers > n_batches:
                    num_workers = n_batches
        else:
            persistent_workers = False

        if num_fetch_workers > 0:
            dl_ctor = partial(ConcurrentDataloader, num_fetch_workers=num_fetch_workers)
        else:
            dl_ctor = DataLoader

        # KDScheduledTransform requires num_workers > 0
        if num_workers == 0:
            multi_view_wrapper = dataset.get_wrapper_of_type(kd.KDMultiViewWrapper)
            if multi_view_wrapper is not None:
                for transform_config in multi_view_wrapper.transform_configs:
                    if isinstance(transform_config.transform, kd.KDScheduledTransform):
                        self.logger.info(f"found KDScheduledTransform with num_workers == 0 -> use num_workers=1")
                        num_workers = 1
            x_transform_wrapper = dataset.get_wrapper_of_type(kd.XTransformWrapper)
            if x_transform_wrapper is not None:
                if isinstance(x_transform_wrapper.transform, kd.KDScheduledTransform):
                    self.logger.info(f"found KDScheduledTransform with num_workers == 0 -> use num_workers=1")
                    num_workers = 1

        dl_kwargs = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            generator=self.generator,
            collate_fn=collator,
        )
        # set worker_init_fn for e.g. KDScheduledTransform
        # don't use this when num_workers=0 as it changes the behavior
        # e.g. the counter of a KDScheduledTransform is with num_workers=0 global
        # specify --num_workers 1 for testing stuff with worker_init_fn
        if end_checkpoint is not None:
            dataset_length = len(dataset)
            if is_distributed():
                dataset_length //= get_world_size()
            dl_kwargs["worker_init_fn"] = partial(
                dataset.worker_init_fn,
                batch_size=batch_size,
                dataset_length=len(dataset),
                drop_last=drop_last,
                epochs=end_checkpoint.epoch,
                updates=end_checkpoint.update,
                samples=end_checkpoint.sample,
            )

        if is_train_dataset:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset, generator=self.generator)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = InfiniteBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
            return dl_ctor(
                dataset=dataset,
                batch_sampler=batch_sampler,
                pin_memory=pin_memory,
                **dl_kwargs,
                **kwargs,
            )

        return dl_ctor(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **dl_kwargs,
            **kwargs,
            pin_memory=pin_memory,
        )

    def dispose(self):
        for dataset in self.datasets.values():
            dataset.dispose()
