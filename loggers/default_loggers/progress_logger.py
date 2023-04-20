from loggers.base.logger_base import LoggerBase


class ProgressLogger(LoggerBase):
    def _log(self, update_counter, **_):
        self.logger.info("------------------")
        self.logger.info(f"Epoch {update_counter.cur_checkpoint.epoch} ({update_counter.cur_checkpoint})")

    def _log_after_update(self, update_counter, **_):
        self.logger.info("------------------")
        self.logger.info(f"Update {update_counter.cur_checkpoint.update} ({update_counter.cur_checkpoint})")

    def _log_after_sample(self, update_counter, **_):
        self.logger.info("------------------")
        self.logger.info(f"Sample {update_counter.cur_checkpoint.sample} ({update_counter.cur_checkpoint})")
