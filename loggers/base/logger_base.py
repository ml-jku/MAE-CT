import logging

import kappaprofiler as kp
import torch

from providers.config_providers.base.config_provider_base import ConfigProviderBase
from providers.config_providers.noop_config_provider import NoopConfigProvider
from providers.stage_path_provider import StagePathProvider
from providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from utils.data_container import DataContainer
from ..writers.checkpoint_writer import CheckpointWriter
from ..writers.noop_writer import NoopWriter


class LoggerBase:
    static_logger = logging.getLogger("LoggerBase")
    log_writer_singleton = None
    noop_writer_singleton = NoopWriter()

    @staticmethod
    def flush():
        if LoggerBase.log_writer_singleton is not None:
            LoggerBase.log_writer_singleton.flush()

    @staticmethod
    def finish():
        if LoggerBase.log_writer_singleton is not None:
            LoggerBase.log_writer_singleton.finish()

    @staticmethod
    def call_before_training(loggers, **kwargs):
        for l in loggers:
            l.before_training(**kwargs)

    @staticmethod
    def call_before_every_update(loggers, **kwargs):
        for l in loggers:
            l.before_every_update(**kwargs)

    @staticmethod
    def call_before_every_accumulation_step(loggers, **kwargs):
        for l in loggers:
            l.before_every_accumulation_step(**kwargs)

    @staticmethod
    def call_start_dataloader_iterators(loggers, **kwargs):
        for l in loggers:
            l.start_dataloader_iterator(**kwargs)

    @staticmethod
    def call_track_after_accumulation_step(loggers, **kwargs):
        for l in loggers:
            l.track_after_accumulation_step(**kwargs)

    @staticmethod
    def call_track_after_update_step(loggers, **kwargs):
        for l in loggers:
            l.track_after_update_step(**kwargs)

    @staticmethod
    def call_after_sample(loggers, **kwargs):
        logger_info_dict = {}
        for l in loggers:
            l.after_sample(logger_info_dict=logger_info_dict, **kwargs)
        return logger_info_dict

    @staticmethod
    def call_after_update(loggers, **kwargs):
        logger_info_dict = {}
        for l in loggers:
            l.after_update(logger_info_dict=logger_info_dict, **kwargs)
        return logger_info_dict

    @staticmethod
    def call_after_epoch(loggers, **kwargs):
        logger_info_dict = {}
        for l in loggers:
            l.after_epoch(logger_info_dict=logger_info_dict, **kwargs)
        return logger_info_dict

    @staticmethod
    def call_single_log(loggers, **kwargs):
        logger_info_dict = {}
        for l in loggers:
            # TODO refactor
            l.single_log(logger_info_dict=logger_info_dict, **kwargs)
        return logger_info_dict

    def __init__(
            self,
            every_n_epochs: int = None,
            every_n_updates: int = None,
            every_n_samples: int = None,
            data_container: DataContainer = None,
            config_provider: ConfigProviderBase = None,
            summary_provider: SummaryProviderBase = None,
            stage_path_provider: StagePathProvider = None,
    ):
        self.data_container = data_container
        self.config_provider = config_provider or NoopConfigProvider()
        self.summary_provider = summary_provider or NoopSummaryProvider()
        self.stage_path_provider = stage_path_provider

        self.every_n_epochs = every_n_epochs
        self.every_n_updates = every_n_updates
        self.every_n_samples = every_n_samples
        if self.allows_no_interval_types:
            assert self.n_interval_types == 0
        else:
            assert self.allows_multiple_interval_types or self.n_interval_types <= 1

        # these things are initialized on property access because they require the name/full_name
        # (which can be set from child classes)
        self._logger = None
        # trainer checkpoint requires gathering random states -> all ranks have a checkpoint writer
        self.checkpoint_writer = CheckpointWriter(stage_path_provider=self.stage_path_provider)

        # check that children only override their implementation methods
        assert type(self).before_training == LoggerBase.before_training
        assert type(self).track_after_accumulation_step == LoggerBase.track_after_accumulation_step
        assert type(self).track_after_update_step == LoggerBase.track_after_update_step
        assert type(self).after_update == LoggerBase.after_update
        assert type(self).after_epoch == LoggerBase.after_epoch
        assert type(self).after_training == LoggerBase.after_training

    @property
    def allows_no_interval_types(self):
        """ some loggers can't make use of any interval type (e.g. DatasetStatsLogger, EtaLogger) """
        return False

    @property
    def allows_multiple_interval_types(self):
        """ loggers that track stuff are inconsistent if multiple interval_types are allowed """
        return True

    def get_updates_per_log_interval(self, update_counter):
        assert not self.allows_multiple_interval_types
        if self.every_n_epochs is not None:
            return update_counter.updates_per_epoch * self.every_n_epochs
        if self.every_n_updates is not None:
            return self.every_n_updates
        if self.every_n_samples is not None:
            # NOTE: uneven every_n_samples not supported
            assert self.every_n_samples % update_counter.effective_batch_size == 0
            return int(self.every_n_samples / update_counter.effective_batch_size)
        raise RuntimeError

    def get_updates_till_next_log(self, update_counter):
        updates_per_log_interval = self.get_updates_per_log_interval(update_counter)
        return updates_per_log_interval - update_counter.cur_checkpoint.update % updates_per_log_interval

    @property
    def n_interval_types(self):
        return sum(
            [self.every_n_epochs is not None, self.every_n_updates is not None, self.every_n_samples is not None])

    @property
    def writer(self):
        if LoggerBase.log_writer_singleton is None:
            return LoggerBase.noop_writer_singleton
        return LoggerBase.log_writer_singleton

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self.name)
        return self._logger

    @property
    def name(self):
        return type(self).__name__

    def _before_training(self, **kwargs):
        pass

    def _after_training(self, **kwargs):
        pass

    def _log_after_sample(self, update_counter, **kwargs):
        self._log(update_counter=update_counter, interval_type="sample", **kwargs)

    def _log_after_update(self, update_counter, **kwargs):
        self._log(update_counter=update_counter, interval_type="update", **kwargs)

    def _log_after_epoch(self, update_counter, **kwargs):
        self._log(update_counter=update_counter, interval_type="epoch", **kwargs)

    def _log(self, update_counter, interval_type, **kwargs):
        pass

    @torch.no_grad()
    def track_after_accumulation_step(self, **kwargs):
        if type(self)._track_after_accumulation_step == LoggerBase._track_after_accumulation_step:
            return
        with kp.named_profile(f"{self.name}.track_after_accumulation_step"):
            self._track_after_accumulation_step(**kwargs)

    @torch.no_grad()
    def track_after_update_step(self, **kwargs):
        if type(self)._track_after_update_step == LoggerBase._track_after_update_step:
            return
        with kp.named_profile(f"{self.name}.track_after_update_step"):
            self._track_after_update_step(**kwargs)

    def _track_after_accumulation_step(self, **kwargs):
        pass

    def _track_after_update_step(self, **kwargs):
        pass

    def before_every_update(self, **kwargs):
        pass

    def before_every_accumulation_step(self, **kwargs):
        pass

    @torch.no_grad()
    def before_training(self, **kwargs):
        if type(self)._before_training == LoggerBase._before_training:
            return
        with kp.named_profile(f"{self.name}.before_training"):
            self._before_training(**kwargs)

    @torch.no_grad()
    def after_training(self, **kwargs):
        if type(self)._after_training == LoggerBase._after_training:
            return

        with kp.named_profile(f"{self.name}.after_training"):
            self._after_training(**kwargs)

    def should_log_after_sample(self, checkpoint, effective_batch_size):
        if self.every_n_samples is not None:
            last_update_samples = checkpoint.sample - effective_batch_size
            prev_log_step = int(last_update_samples / self.every_n_samples)
            cur_log_step = int(checkpoint.sample / self.every_n_samples)
            if cur_log_step > prev_log_step:
                return True
        return False

    @torch.no_grad()
    def after_update(self, update_counter, effective_batch_size, **kwargs):
        overrides_log = type(self)._log == LoggerBase._log
        if overrides_log or type(self)._log_after_sample == LoggerBase._log_after_sample:
            if self.should_log_after_sample(update_counter.cur_checkpoint, effective_batch_size):
                with kp.named_profile(f"{self.name}.after_sample"):
                    self._log_after_sample(update_counter, **kwargs)
        if overrides_log or type(self)._log_after_update == LoggerBase._log_after_update:
            if self.should_log_after_update(update_counter.cur_checkpoint):
                with kp.named_profile(f"{self.name}.after_update"):
                    self._log_after_update(update_counter, **kwargs)

    def should_log_after_update(self, checkpoint):
        if self.every_n_updates is not None:
            return checkpoint.update % self.every_n_updates == 0
        return False

    def should_log_after_epoch(self, checkpoint):
        if self.every_n_epochs is not None:
            return checkpoint.epoch % self.every_n_epochs == 0
        return False

    @torch.no_grad()
    def after_epoch(self, update_counter, **kwargs):
        if type(self)._log == LoggerBase._log and type(self)._log_after_epoch == LoggerBase._log_after_epoch:
            return
        if self.should_log_after_epoch(update_counter.cur_checkpoint):
            with kp.named_profile(f"{self.name}.after_epoch"):
                self._log_after_epoch(update_counter, **kwargs)

    @torch.no_grad()
    def single_log(self, **kwargs):
        self._log(interval_type="epoch", **kwargs)

    @staticmethod
    def epoch_str(epoch):
        return thousand_seperated_int(epoch, min_digits=5, padding_char='X')

    @staticmethod
    def update_str(update):
        return thousand_seperated_int(update, min_digits=7, padding_char='X')

    def to_verbose_interval_string(self):
        results = []
        if self.every_n_epochs is not None:
            results.append(f"every_n_epochs={self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"every_n_updates={self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"every_n_samples={self.every_n_samples}")
        return ", ".join(results)

    def to_short_interval_string(self):
        results = []
        if self.every_n_epochs is not None:
            results.append(f"E{self.every_n_epochs}")
        if self.every_n_updates is not None:
            results.append(f"U{self.every_n_updates}")
        if self.every_n_samples is not None:
            results.append(f"S{self.every_n_samples}")
        return "_".join(results)

    def start_dataloader_iterator(self, **_):
        pass
