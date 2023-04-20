from .logger_base import LoggerBase


class SummaryLogger(LoggerBase):
    def __init__(self, every_n_epochs=None, every_n_updates=None, every_n_samples=None, **kwargs):
        assert every_n_epochs is None
        assert every_n_updates is None
        assert every_n_samples is None
        super().__init__(**kwargs)
