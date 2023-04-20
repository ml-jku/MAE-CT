import yaml

from .base.config_provider_base import ConfigProviderBase
from ..stage_path_provider import StagePathProvider


class PrimitiveConfigProvider(ConfigProviderBase):
    def __init__(self, stage_path_provider: StagePathProvider):
        super().__init__()
        self.stage_path_provider = stage_path_provider
        self.config = {}

    def update(self, *args, **kwargs):
        self.config.update(*args, **kwargs)
        self._save_config_as_yaml()

    def __setitem__(self, key, value):
        self.config[key] = value
        self._save_config_as_yaml()

    def __contains__(self, key):
        return key in self.config

    def get_config_of_previous_stage(self, stage_name, stage_id):
        config_uri = self.stage_path_provider.get_primitive_config_uri(stage_name=stage_name, stage_id=stage_id)
        if not config_uri.exists():
            return None

        with open(config_uri) as f:
            return yaml.safe_load(f)

    def _save_config_as_yaml(self):
        with open(self.stage_path_provider.primitive_config_uri, "w") as f:
            yaml.safe_dump(dict(self.config.items()), f)
