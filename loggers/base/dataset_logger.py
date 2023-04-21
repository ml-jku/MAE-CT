import kappaprofiler as kp
import numpy as np
import torch

from distributed.gather import all_gather_nograd
from distributed.gather import all_gather_nograd_clipped
from utils.formatting_util import list_to_string
from utils.naming_util import snake_type_name
from .logger_base import LoggerBase


class DatasetLogger(LoggerBase):
    def __init__(self, dataset_key, max_size=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.max_size = max_size
        self.total_iter_time = 0.
        self.total_data_time = 0.
        self.total_forward_time = 0.
        # dataloader starts prefetching once iter is called -> call during epoch to have data ready when log is called
        self.loader = None
        self.iterator = None
        self.iterator_stopwatch = None

        # check that children only override their implementation methods
        assert type(self)._before_training == DatasetLogger._before_training

    def _before_training(self, trainer, trainer_batch_size, **kwargs):
        self.loader = self.data_container.dataloader_from_key(
            is_train_dataset=False,
            dataset_key=self.dataset_key,
            mode=self.get_dataset_mode(trainer),
            return_ctx=self.return_ctx,
            batch_size=trainer_batch_size,
            max_size=self.max_size,
        )
        self._before_training_impl(**kwargs)

    def _before_training_impl(self, **kwargs):
        pass

    def start_dataloader_iterator(self, **_):
        if self.iterator is not None:
            # iterator wasn't consumed yet
            return
        with kp.Stopwatch() as iter_stopwatch:
            self.iterator = iter(self.loader)
            self.logger.info(f"started dataloader iterator of {type(self).__name__}(dataset_key={self.dataset_key})")
        self.iterator_stopwatch = iter_stopwatch

    def get_dataset_mode(self, trainer):
        raise NotImplementedError

    @property
    def return_ctx(self):
        return False

    # TODO replace calls to this with the auto-collated version
    def iterate_over_dataset(self, forward_fn, update_counter):
        data_times = []
        forward_times = []
        forward_results = []
        iter_time = self.iterator_stopwatch.elapsed_seconds
        while True:
            # load data
            with kp.Stopwatch() as sw:
                batch = next(self.iterator, None)
            if batch is None:
                self.iterator = None
                self.iterator_stopwatch = None
                break
            # forward
            data_times.append(sw.elapsed_seconds)
            with kp.Stopwatch() as sw:
                forward_result = forward_fn(batch)
            forward_results.append(forward_result)
            forward_times.append(sw.elapsed_seconds)
        # profiling bookkeeping
        mean_data_time = float(np.mean(data_times))
        mean_forward_time = float(np.mean(forward_times))
        stdout_prefix = f"{snake_type_name(self)}_{self.dataset_key}"
        self.logger.info(
            f"{stdout_prefix}_iter={iter_time:.2f} "
            f"{stdout_prefix}_data={mean_data_time:.2f} "
            f"{stdout_prefix}_forward={mean_forward_time:.2f}"
        )
        key_prefix = f"profiling/{snake_type_name(self)}/{self.dataset_key}"
        self.writer.add_scalar(f"{key_prefix}/iter_time", iter_time, update_counter)
        self.writer.add_scalar(f"{key_prefix}/data_time", mean_data_time, update_counter)
        self.writer.add_scalar(f"{key_prefix}/forward_times", mean_forward_time, update_counter)
        self.total_iter_time += iter_time
        self.total_data_time += mean_data_time
        self.total_forward_time += mean_forward_time
        return forward_results

    def _collate_result(self, result):
        if isinstance(result[0], dict):
            # tuple[dict] -> dict[tensor]
            result = {k: torch.concat([r[k] for r in result]) for k in result[0].keys()}
            # gather
            result = {k: all_gather_nograd_clipped(v, len(self.dataset)) for k, v in result.items()}
        else:
            # list -> tensor
            result = torch.concat(result)
            # gather
            result = all_gather_nograd_clipped(result, len(self.dataset))
        return result

    def iterate_over_dataset_collated(self, forward_fn, update_counter):
        forward_results = self.iterate_over_dataset(forward_fn=forward_fn, update_counter=update_counter)
        if isinstance(forward_results[0], tuple):
            return [self._collate_result(result) for result in zip(*forward_results)]
        else:
            return self._collate_result(forward_results)

    @property
    def dataset(self):
        return self.data_container.get_dataset(self.dataset_key)

    def _after_training(self, **_):
        total_iter_time = all_gather_nograd(self.total_iter_time)
        total_data_time = all_gather_nograd(self.total_data_time)
        total_forward_time = all_gather_nograd(self.total_forward_time)
        self.logger.info("------------------")
        self.logger.info(f"{snake_type_name(self)} dataset_key={self.dataset_key}")
        self.logger.info(f"total_iter:   {list_to_string(total_iter_time)}")
        self.logger.info(f"total_data_time:    {list_to_string(total_data_time)}")
        self.logger.info(f"total_forward_time: {list_to_string(total_forward_time)}")
