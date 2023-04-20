from collections import defaultdict

import numpy as np

from distributed.gather import all_reduce_mean_grad
from loggers.base.logger_base import LoggerBase


class OnlineLossLogger(LoggerBase):
    def __init__(self, verbose, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.tracked_losses = defaultdict(list)

    def _track_after_accumulation_step(self, losses, **kwargs):
        for name, loss in losses.items():
            self.tracked_losses[name].append(loss.item())

    @property
    def allows_multiple_interval_types(self):
        return False

    def _log(self, update_counter, **_):
        for name, tracked_loss in self.tracked_losses.items():
            mean_loss = np.mean(tracked_loss)
            mean_loss = all_reduce_mean_grad(mean_loss)
            self.writer.add_scalar(f"loss/online/{name}/{self.to_short_interval_string()}", mean_loss, update_counter)
            if self.verbose:
                # log the average loss since the last log
                self.logger.info(f"loss/online/{name}: {mean_loss:.8f}")

        self.tracked_losses.clear()
