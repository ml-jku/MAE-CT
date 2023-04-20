import logging

from providers.config_providers.base.config_provider_base import ConfigProviderBase
from providers.config_providers.noop_config_provider import NoopConfigProvider
from providers.stage_path_provider import StagePathProvider
from providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from utils.data_container import DataContainer


class TrainerInterface:
    def __init__(
            self,
            data_container: DataContainer,
            config_provider: ConfigProviderBase = None,
            summary_provider: SummaryProviderBase = None,
            stage_path_provider: StagePathProvider = None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.data_container = data_container
        self.config_provider = config_provider or NoopConfigProvider()
        self.summary_provider = summary_provider or NoopSummaryProvider()
        self.stage_path_provider = stage_path_provider

    def train(self, model):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    @property
    def dataset_mode(self):
        raise NotImplementedError
