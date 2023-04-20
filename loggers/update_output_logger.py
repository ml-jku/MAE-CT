import numpy as np
import torch

from distributed.gather import all_reduce_mean_grad
from loggers.base.logger_base import LoggerBase


class UpdateOutputLogger(LoggerBase):
    def __init__(self, key, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.tracked_values = []
        self.key = key
        if self.every_n_updates == 1:
            verbose = False
        self.verbose = verbose

    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        value = update_outputs[self.key]
        if torch.is_tensor(value):
            if value.numel() > 1:
                value = value.mean()
            value = value.item()
        self.tracked_values.append(value)

    @property
    def allows_multiple_interval_types(self):
        return False

    def _log(self, update_counter, **_):
        mean_value = np.mean(self.tracked_values)
        mean_value = all_reduce_mean_grad(mean_value)
        if self.verbose:
            self.logger.info(f"{self.key}: {mean_value:.5f}")
        # change order of key (e.g. key="mocov3/nn_accuracy" -> "nn_accuracy/mocov3")
        inverted_key = "/".join(reversed(self.key.split("/"))).replace(".", "/")
        # use to_short_interval_string here because this is basically an online logger
        self.writer.add_scalar(f"{inverted_key}/{self.to_short_interval_string()}", mean_value, update_counter)
        self.tracked_values.clear()
