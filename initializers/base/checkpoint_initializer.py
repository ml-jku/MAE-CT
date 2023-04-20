import torch
import yaml

from initializers.base.initializer_base import InitializerBase
from models.base.single_model_base import SingleModelBase
from providers.config_providers.base.config_provider_base import ConfigProviderBase
from providers.summary_providers.base.summary_provider_base import SummaryProviderBase
from utils.checkpoint import Checkpoint
from utils.factory import create


class CheckpointInitializer(InitializerBase):
    def __init__(self, stage_id, model_name, checkpoint, load_optim, model_info=None, stage_name=None, **kwargs):
        super().__init__(**kwargs)
        self.stage_id = stage_id
        self.model_name = model_name
        self.load_optim = load_optim
        self.model_info = model_info
        self.stage_name = stage_name or self.stage_path_provider.stage_name

        # checkpoint can be a string (e.g. "best_accuracy" for initializing from a model saved by BestModelLogger)
        # or dictionary with epoch/update/sample values
        if isinstance(checkpoint, str):
            self.checkpoint = checkpoint
        else:
            self.checkpoint = create(checkpoint, Checkpoint)
            assert self.checkpoint.is_minimally_specified or self.checkpoint.is_fully_specified

    @property
    def should_apply_model_specific_initialization(self):
        # applying model specific initialization would overwrite the loaded weights with random weights
        return False

    def init_weights(self, model, config_provider=None, summary_provider=None):
        sd, model_name, ckpt_uri = self.get_model_state_dict(model)
        model.load_state_dict(sd)
        self.logger.info(f"loaded weights of {model_name} from {ckpt_uri}")
        self._copy_config_and_summary(config_provider=config_provider, summary_provider=summary_provider)

    def get_model_state_dict(self, model):
        model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, file_type="model")
        sd = torch.load(ckpt_uri, map_location=model.device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        return sd, model_name, ckpt_uri

    def init_optim(self, model):
        if not isinstance(model, SingleModelBase):
            return
        if not self.load_optim:
            return
        if model.optim is None:
            # e.g. EMA target network doesn't have an optimizer
            model_name = self.model_name or model.name
            self.logger.info(f"skipping loading optimizer for model {model_name} (optimizer is None)")
            return
        model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, file_type="optim")
        sd = torch.load(ckpt_uri, map_location=model.device)
        model.optim.load_state_dict(sd)
        self.logger.info(f"loaded optimizer of {model_name} from {ckpt_uri}")

    def _get_modelname_and_ckpturi(self, model, file_type, model_name=None):
        assert isinstance(model, SingleModelBase)
        model_name = model_name or self.model_name
        if model_name is None:
            self.logger.info(f"no model_name provided -> using {model.name}")
            model_name = model.name

        # model_type is e.g. ema=0.99
        model_info_str = "" if self.model_info is None else f" {self.model_info}"
        ckpt_uri = self._get_ckpt_uri(prefix=f"{model_name} cp=", suffix=f"{model_info_str} {file_type}.th")
        # LEGACY previously models stored last checkpoint without cp= prefix
        if not ckpt_uri.exists():
            legacy_ckpt_uri = self._get_ckpt_uri(prefix=f"{model_name} ", suffix=f" {file_type}.th")
            if legacy_ckpt_uri.exists():
                ckpt_uri = legacy_ckpt_uri
        assert ckpt_uri.exists(), f"{ckpt_uri} doesn't exist"
        return model_name, ckpt_uri

    def _get_ckpt_uri(self, prefix, suffix):
        ckpt_folder = self.stage_path_provider.get_stage_checkpoint_path(
            stage_name=self.stage_name,
            stage_id=self.stage_id,
        )
        # find full checkpoint from minimal specification
        if not isinstance(self.checkpoint, str) and not self.checkpoint.is_fully_specified:
            ckpt = Checkpoint.to_fully_specified_from_fnames(
                ckpt_folder=ckpt_folder,
                ckpt=self.checkpoint,
                prefix=prefix,
                suffix=suffix,
            )
        else:
            ckpt = self.checkpoint
        return ckpt_folder / f"{prefix}{ckpt}{suffix}"

    def _copy_config_and_summary(self, config_provider: ConfigProviderBase, summary_provider: SummaryProviderBase):
        self.logger.info(f"copying config and summary")
        # add config from previous stage
        if config_provider is not None:
            prev_stage_config = config_provider.get_config_of_previous_stage(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
            )
            if prev_stage_config is None:
                self.logger.info(f"prev_stage_config is None -> don't copy anything")
            else:
                # exclude irrelevant stuff (e.g. device or dataloader params are irrelevant)
                def include(key):
                    # accessible via initializer
                    if key in ["run_name", "stage_name"]:
                        return False
                    # dependent on the hardware which produced the checkpoint
                    if key in ["device", "trainer/accumulation_steps", "trainer/batch_size"]:
                        return False
                    if key.startswith("dataloader/") or key.startswith("dist/"):
                        return False
                    return True

                prev_stage_config = {k: v for k, v in prev_stage_config.items() if include(k)}
                if self.stage_name not in config_provider:
                    config_provider[self.stage_name] = prev_stage_config
        else:
            self.logger.info(f"no config_provider -> can't copy config")

        # add summary from previous stage
        if summary_provider is not None:
            prev_stage_summary = summary_provider.get_summary_of_previous_stage(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
            )
            if prev_stage_summary is None:
                self.logger.info(f"prev_stage_summary is None -> don't copy anything")
            else:
                # exclude irrelevant stuff (e.g. profiling times)
                def include(key):
                    if key.startswith("profiler/") or key.startswith("profiling/") or key.startswith("lr/"):
                        return False
                    return True

                prev_stage_summary = {k: v for k, v in prev_stage_summary.items() if include(k)}
                for prev_key, prev_value in prev_stage_summary.items():
                    new_key = f"{self.stage_name}/{prev_key}"
                    if new_key not in summary_provider:
                        summary_provider[new_key] = prev_value
        else:
            self.logger.info(f"no summary_provider -> can't copy summary")

    def get_kwargs(self):
        ckpt_path = self.stage_path_provider.get_stage_checkpoint_path(
            stage_name=self.stage_name,
            stage_id=self.stage_id,
        )
        with open(ckpt_path / f"{self.model_name} kwargs.yaml") as f:
            return yaml.safe_load(f)
