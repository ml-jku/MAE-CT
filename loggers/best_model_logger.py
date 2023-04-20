from loggers.base.logger_base import LoggerBase
from utils.infer_higher_is_better import higher_is_better_from_metric_key


class BestModelLogger(LoggerBase):
    def __init__(self, metric_key, tolerances=None, model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.metric_key = metric_key
        self.model_name = model_name
        self.higher_is_better = higher_is_better_from_metric_key(self.metric_key)
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")

        # save multiple best models based on tolerance
        self.tolerances_is_exceeded = {tolerance: False for tolerance in tolerances or []}
        self.tolerance_counter = 0
        self.metric_at_exceeded_tolerance = {}

    def _before_training(self, update_counter, **kwargs):
        if len(self.tolerances_is_exceeded) > 0:
            if update_counter.cur_checkpoint.sample > 0:
                raise NotImplementedError("BestModelLogger with tolerances resuming not implemented")

    def _extract_metric_value(self, logger_info_dict):
        metric_value = logger_info_dict.get(self.metric_key, None)
        assert metric_value is not None, (
            f"couldn't find metric_value {self.metric_key} (valid metric keys={list(logger_info_dict.keys())}). "
            f"make sure logger that produces the metric key is called beforehand"
        )
        return metric_value

    def _is_new_best_model(self, metric_value):
        if self.higher_is_better:
            return metric_value > self.best_metric_value
        else:
            return metric_value < self.best_metric_value

    # noinspection PyMethodOverriding
    def _log(self, update_counter, trainer, model, logger_info_dict, **kwargs):
        metric_value = self._extract_metric_value(logger_info_dict)
        if self._is_new_best_model(metric_value):
            # one could also track the model and save it after training
            # this is better in case runs crash or are terminated
            # the runtime overhead is neglegible
            self.logger.info(f"new best model ({self.metric_key}): {self.best_metric_value} --> {metric_value}")
            self.checkpoint_writer.save(
                model=model,
                checkpoint=f"best_model.{self.metric_key.replace('/', '.')}",
                save_mode="seperate",
                save_optim=False,
                model_name_to_save=self.model_name,
            )
            self.best_metric_value = metric_value
            self.tolerance_counter = 0
            # log tolerance checkpoints
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                self.checkpoint_writer.save(
                    model=model,
                    checkpoint=f"best_model.{self.metric_key.replace('/', '.')}.tolerance{tolerance}",
                    save_mode="seperate",
                    save_optim=False,
                    model_name_to_save=self.model_name,
                )
        else:
            self.tolerance_counter += 1
            for tolerance, is_exceeded in self.tolerances_is_exceeded.items():
                if is_exceeded:
                    continue
                if tolerance >= self.tolerance_counter:
                    self.tolerances_is_exceeded[tolerance] = True
                    self.metric_at_exceeded_tolerance[tolerance] = metric_value

    def _after_training(self, **kwargs):
        # best metric doesn't need to be logged as it is summarized anyways
        for tolerance, value in self.metric_at_exceeded_tolerance.items():
            self.logger.info(f"best {self.metric_key} with tolerance={tolerance}: {value}")
