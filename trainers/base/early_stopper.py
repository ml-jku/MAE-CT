import logging

from utils.infer_higher_is_better import higher_is_better_from_metric_key
from utils.param_checking import check_exclusive


class EarlyStopper:
    _INVALID_METRIC_ERROR = \
        "Couldn't find metric_key {} (valid metric_keys={}). Make sure every_n_epochs/every_n_updates/every_n_samples" \
        " is aligned with the corresponding logger."
    _INVALID_INTERVAL_ERROR = "specify only one of every_n_epochs/every_n_updates/every_n_samples"

    def __init__(
            self,
            metric_key,
            tolerance,
            every_n_epochs=None,
            every_n_updates=None,
            every_n_samples=None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better_from_metric_key(self.metric_key)
        assert check_exclusive(every_n_epochs, every_n_updates, every_n_samples), self._INVALID_INTERVAL_ERROR
        assert tolerance is not None and tolerance >= 1, "tolerance has to be >= 1"
        self.every_n_epochs = every_n_epochs
        self.every_n_updates = every_n_updates
        self.every_n_samples = every_n_samples
        self.tolerance = tolerance
        self.tolerance_counter = 0
        self.best_metric = -float("inf") if self.higher_is_better else float("inf")

    def should_stop_after_sample(self, checkpoint, logger_info_dict, effective_batch_size):
        if self.every_n_samples is not None:
            last_update_samples = checkpoint.sample - effective_batch_size
            prev_log_step = int(last_update_samples / self.every_n_samples)
            cur_log_step = int(checkpoint.sample / self.every_n_samples)
            if cur_log_step > prev_log_step:
                return self._should_stop(logger_info_dict)
        return False

    def should_stop_after_update(self, checkpoint, logger_info_dict):
        if self.every_n_updates is None or checkpoint.update % self.every_n_updates != 0:
            return False
        return self._should_stop(logger_info_dict)

    def should_stop_after_epoch(self, checkpoint, logger_info_dict):
        if self.every_n_epochs is None or checkpoint.epoch % self.every_n_epochs != 0:
            return False
        return self._should_stop(logger_info_dict)

    def _metric_improved(self, cur_metric):
        if self.higher_is_better:
            return cur_metric > self.best_metric
        return cur_metric < self.best_metric

    def _should_stop(self, logger_info_dict):
        cur_metric = logger_info_dict.get(self.metric_key, None)
        assert cur_metric is not None, self._INVALID_METRIC_ERROR.format(self.metric_key, list(logger_info_dict.keys()))

        if self._metric_improved(cur_metric):
            self.logger.info(f"{self.metric_key} improved: {self.best_metric} --> {cur_metric}")
            self.best_metric = cur_metric
            self.tolerance_counter = 0
        else:
            self.tolerance_counter += 1
            cmp_str = "<=" if self.higher_is_better else ">="
            stop_training_str = " --> stop training" if self.tolerance_counter >= self.tolerance else ""
            self.logger.info(
                f"{self.metric_key} stagnated: {self.best_metric} {cmp_str} {cur_metric} "
                f"({self.tolerance_counter}/{self.tolerance}){stop_training_str}"
            )

        return self.tolerance_counter >= self.tolerance
