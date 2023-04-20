import torch

from models.base.composite_model_base import CompositeModelBase
from models.base.single_model_base import SingleModelBase
from .base.checkpoint_initializer import CheckpointInitializer


class ResumeInitializer(CheckpointInitializer):
    """
    initializes models/optims from a checkpoint ready for resuming training
    load_optim=True as this is usually used to resume a training run
    stage_name is provided by the trainer as it already knows the correct stage_name
    """

    def __init__(self, load_optim=True, load_random_states=True, **kwargs):
        super().__init__(load_optim=load_optim, model_name=None, **kwargs)
        # TODO pull Epoch/Update/Sample information from trainer checkpoint if checkpoint is a string
        self.load_random_states = load_random_states

    # TODO ugly implementation
    def init_weights(self, model, **_):
        self._init_weights(model.name, model)

    def _init_weights(self, name, model):
        if isinstance(model, SingleModelBase):
            if model.is_frozen:
                self.logger.info(
                    f"skip loading weights from checkpoint '{self.checkpoint}' for {model.name} "
                    f"(is_frozen)"
                )
            else:
                model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, model_name=name, file_type="model")
                sd = torch.load(ckpt_uri, map_location=model.device)
                if "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd)
                self.logger.info(f"loaded weights of {model_name} from {ckpt_uri}")
        if isinstance(model, CompositeModelBase):
            for submodel_name, submodel in model.submodels.items():
                self._init_weights(name=f"{name}.{submodel_name}", model=submodel)

    # TODO ugly implementation
    def init_optim(self, model):
        self._init_optim(name=model.name, model=model)

    def _init_optim(self, name, model):
        if isinstance(model, SingleModelBase):
            if model.optim is None:
                # e.g. EMA target network doesn't have an optimizer
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint}' for {model.name} "
                    f"(optim is None)"
                )
            if model.is_frozen:
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint}' for {model.name} "
                    f"(is_frozen)"
                )
            else:
                model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, model_name=name, file_type="optim")
                sd = torch.load(ckpt_uri, map_location=model.device)
                model.optim.load_state_dict(sd)
                self.logger.info(f"loaded optimizer of {model_name} from {ckpt_uri}")
        if isinstance(model, CompositeModelBase):
            for submodel_name, submodel in model.submodels.items():
                self._init_optim(name=f"{name}.{submodel_name}", model=submodel)

    def init_trainer(self, trainer):
        # LEGACY: checkpoints before 27.10.2022 don't have a trainer checkpoint
        try:
            ckpt_uri = self._get_ckpt_uri(prefix=f"trainer cp=", suffix=".th")
        except FileNotFoundError:
            self.logger.warning(f"no trainer checkpoint found for checkpoint {checkpoint}")
            return
        trainer.load_state_dict(torch.load(ckpt_uri), load_random_states=self.load_random_states)
        self.logger.info(f"loaded trainer checkpoint {ckpt_uri}")
