import logging

from providers.stage_path_provider import StagePathProvider


class InitializerBase:
    def __init__(self, stage_path_provider: StagePathProvider = None):
        self.logger = logging.getLogger(type(self).__name__)
        self.stage_path_provider = stage_path_provider

    def init_weights(self, model, config_provider=None, summary_provider=None):
        raise NotImplementedError

    def init_optim(self, model):
        pass

    @property
    def should_apply_model_specific_initialization(self):
        # whether or not model specific intialization is applied after the initializer
        raise NotImplementedError
