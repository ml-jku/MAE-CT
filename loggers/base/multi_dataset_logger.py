from collections import defaultdict

import kappaprofiler as kp
import numpy as np
import torch
from tqdm import tqdm

from distributed.config import is_managed
from distributed.gather import all_gather_nograd
from distributed.gather import all_gather_nograd_clipped
from utils.formatting_util import list_to_string
from utils.naming_util import snake_type_name
from utils.noop_tqdm import NoopTqdm
from .logger_base import LoggerBase


# TODO MultiDatasetLogger and DatasetLogger are very similar (MultiDatasetLogger should be able to disable prefetching (e.g. for EmbeddingProjectorLogger)
class MultiDatasetLogger(LoggerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_iter_time = defaultdict(float)
        self.total_data_time = defaultdict(float)
        self.total_forward_time = defaultdict(float)

    def iterate_over_dataset(
            self,
            forward_fn,
            batch_size,
            dataset_key,
            dataset_mode,
            update_counter=None,
            max_size=None,
            shuffle_seed=None,
            persistent_workers=None,
            use_collate_fn=True
    ):
        # TODO for some reason this hands when datasets are iterated in immediate succession
        #  maybe this is because the workers from the train iteration are not killed yet and it somehow kills
        #  a testset worker...no ide why it happens but a timeout solves this issue
        from time import sleep
        sleep(1)
        # init loader
        loader = self.data_container.dataloader_from_key(
            is_train_dataset=False,
            dataset_key=dataset_key,
            mode=dataset_mode,
            return_ctx=True,
            batch_size=batch_size,
            max_size=max_size,
            shuffle_seed=shuffle_seed,
            persistent_workers=persistent_workers,
        )
        with kp.Stopwatch() as sw:
            iterator = iter(loader)
        iter_time = sw.elapsed_seconds

        # iterate
        data_times = []
        forward_times = []
        forward_results = []
        with NoopTqdm() if is_managed() else tqdm(total=len(loader)) as pbar:
            while True:
                # load data
                with kp.Stopwatch() as data_sw:
                    batch = next(iterator, None)
                if batch is None:
                    break
                data_times.append(data_sw.elapsed_seconds)
                # forward
                with kp.Stopwatch() as forward_sw:
                    forward_result = forward_fn(batch)
                forward_times.append(forward_sw.elapsed_seconds)
                forward_results.append(forward_result)
                pbar.update(1)

        # profiling book keeping
        mean_data_time = float(np.mean(data_times))
        mean_forward_time = float(np.mean(forward_times))
        prefix = f"profiling/{snake_type_name(self)}/{dataset_key}"
        self.logger.info(f"{prefix}: iter={iter_time:.2f} data={mean_data_time:.2f} forward={mean_forward_time:.2f}")
        if update_counter is not None:
            self.writer.add_scalar(f"{prefix}/iter_time", iter_time, update_counter)
            self.writer.add_scalar(f"{prefix}/data_time", mean_data_time, update_counter)
            self.writer.add_scalar(f"{prefix}/forward_times", mean_forward_time, update_counter)
        self.total_iter_time[dataset_key] += iter_time
        self.total_data_time[dataset_key] += mean_data_time
        self.total_forward_time[dataset_key] += mean_forward_time

        # collate
        if use_collate_fn:
            single_output = False
            if not isinstance(forward_results[0], tuple):
                forward_results = [(fwr,) for fwr in forward_results]
                single_output = True
            collated = [
                self._collate_result(result, dataset_len=len(loader.dataset))
                for result in zip(*forward_results)
            ]

            if single_output:
                return collated[0]
        else:
            collated = forward_results

        return collated

    @staticmethod
    def _collate_result(result, dataset_len):
        if isinstance(result[0], dict):
            # tuple[dict] -> dict[tensor]
            result = {k: torch.concat([r[k] for r in result]) for k in result[0].keys()}
            # gather
            result = {k: all_gather_nograd_clipped(v, dataset_len) for k, v in result.items()}
        else:
            if isinstance(result[0], list):
                # List[List[Tensor]] -> List[Tensor]
                result = [torch.concat(item) for item in zip(*result)]
                result = [all_gather_nograd_clipped(item, dataset_len) for item in result]
            else:
                # List[Tensor] -> Tensor
                result = torch.concat(result)
                result = all_gather_nograd_clipped(result, dataset_len)
        return result

    def _after_training(self, **_):
        for dataset_key in self.total_iter_time.keys():
            total_iter_time = all_gather_nograd(self.total_iter_time[dataset_key])
            total_data_time = all_gather_nograd(self.total_data_time[dataset_key])
            total_forward_time = all_gather_nograd(self.total_forward_time[dataset_key])
            self.logger.info("------------------")
            self.logger.info(f"{snake_type_name(self)} dataset_key={dataset_key}")
            self.logger.info(f"total_iter:   {list_to_string(total_iter_time)}")
            self.logger.info(f"total_data_time:    {list_to_string(total_data_time)}")
            self.logger.info(f"total_forward_time: {list_to_string(total_forward_time)}")
