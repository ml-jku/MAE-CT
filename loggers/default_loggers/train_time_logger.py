import numpy as np

from distributed.gather import all_gather_nograd
from loggers.base.logger_base import LoggerBase
from utils.formatting_util import list_to_string


class TrainTimeLogger(LoggerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_times = []
        self.update_times = []
        self.iter_times = []
        self.total_iter_time = 0.
        self.total_data_time = 0.
        self.total_update_time = 0.

    def _track_after_update_step(self, **kwargs):
        times = kwargs["times"]
        iter_time = times["iter_time"]
        if iter_time is not None:
            self.iter_times.append(iter_time)
        self.data_times.append(times["data_time"])
        self.update_times.append(times["update_time"])

    @property
    def allows_multiple_interval_types(self):
        return False

    def _log(self, update_counter, interval_type, **_):
        mean_data_time = np.mean(self.data_times)
        mean_upd_time = np.mean(self.update_times)
        self.total_data_time += mean_data_time * len(self.data_times)
        self.total_update_time += mean_upd_time * len(self.update_times)
        self.data_times = []
        self.update_times = []

        # gather for all devices
        mean_data_times = all_gather_nograd(mean_data_time)
        mean_update_times = all_gather_nograd(mean_upd_time)

        for i, (mean_data_time, mean_upd_time) in enumerate(zip(mean_data_times, mean_update_times)):
            # wandb doesn't like it when system/<key> values are logged
            self.writer.add_scalar(f"profiling/trainer/data_time/{i}/{interval_type}", mean_data_time, update_counter)
            self.writer.add_scalar(f"profiling/trainer/update_time/{i}/{interval_type}", mean_upd_time, update_counter)

        if len(self.iter_times) > 0:
            mean_iter_time = np.mean(self.iter_times)
            self.total_iter_time += mean_iter_time * len(self.iter_times)
            self.iter_times = []
            mean_iter_times = all_gather_nograd(mean_iter_time)
            for i, mean_iter_time in enumerate(mean_iter_times):
                # NOTE if multiple interval_types or multiple TrainTimeLoggers are used this will be called multiple
                # times with the same key which will probably mess with wandb visualization
                self.writer.add_scalar(f"profiling/trainer/iter_time/{i}", mean_iter_time, update_counter)
            iter_times_str = list_to_string(mean_iter_times)
        else:
            iter_times_str = None
        mean_data_times_str = list_to_string(mean_data_times)
        mean_upd_times_str = list_to_string(mean_update_times)
        if iter_times_str is not None:
            self.logger.info(f"train_iter={iter_times_str} train_data={mean_data_times_str} train={mean_upd_times_str}")
        else:
            self.logger.info(f"train_data={mean_data_times_str} train={mean_upd_times_str}")

    def _after_training(self, update_counter, **_):
        total_iter_time = all_gather_nograd(self.total_iter_time)
        total_data_time = all_gather_nograd(self.total_data_time)
        total_update_time = all_gather_nograd(self.total_update_time)
        self.logger.info("------------------")
        self.logger.info(f"total_train_iter:  {list_to_string(total_iter_time)}")
        self.logger.info(f"total_data_time:   {list_to_string(total_data_time)}")
        self.logger.info(f"total_update_time: {list_to_string(total_update_time)}")
