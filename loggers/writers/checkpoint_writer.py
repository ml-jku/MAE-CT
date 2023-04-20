import logging

import torch
import yaml

from distributed.config import is_rank0
from distributed.distributed_data_parallel import DistributedDataParallel
from models.base.composite_model_base import CompositeModelBase
from models.base.single_model_base import SingleModelBase
from providers.stage_path_provider import StagePathProvider


class CheckpointWriter:
    SAVE_MODES = ["all", "seperate", "single"]

    def __init__(self, stage_path_provider: StagePathProvider):
        self.logger = logging.getLogger(type(self).__name__)
        self.stage_path_provider = stage_path_provider

    @staticmethod
    def _unwrapp_model(model):
        # DistributedDataParallel wraps model which would add a 'module.' prefix to every key a saved state_dict
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    def save(
            self,
            model,
            checkpoint,
            save_mode="seperate",
            trainer=None,
            save_weights=True,
            save_optim=True,
            save_latest_weights=False,
            save_latest_optim=False,
            model_name_to_save=None,
            save_frozen_weights=False,
    ):
        assert save_mode in self.SAVE_MODES
        # NOTE: this has to be called from all ranks because random states are gathered to rank0
        trainer_sd = trainer.state_dict() if trainer is not None else None
        if is_rank0():
            if save_mode in ["all", "seperate"]:
                self._save_seperate_models(
                    name=model.name,
                    model=model,
                    ckpt=checkpoint,
                    save_weights=save_weights,
                    save_optim=save_optim,
                    save_latest_weights=save_latest_weights,
                    save_latest_optim=save_latest_optim,
                    model_name_to_save=model_name_to_save,
                    save_frozen_weights=save_frozen_weights,
                )
                trainer_out_path = self.stage_path_provider.checkpoint_path / f"trainer cp={checkpoint}.th"
                if trainer_sd is not None:
                    torch.save(trainer_sd, trainer_out_path)
                self.logger.info(f"saved trainer state_dict to {trainer_out_path}")
            if save_mode in ["all", "single"]:
                assert model_name_to_save is None
                self._save_single(
                    trainer_sd=trainer_sd,
                    model=model,
                    ckpt=checkpoint,
                    save_weights=save_weights,
                    save_optim=save_optim,
                )

    @property
    def _out_path(self):
        return self.stage_path_provider.checkpoint_path

    def _save_seperate_models(
            self,
            name,
            model,
            ckpt,
            save_weights,
            save_optim,
            save_latest_weights,
            save_latest_optim,
            model_name_to_save,
            save_frozen_weights,
    ):
        model = self._unwrapp_model(model)
        if isinstance(model, SingleModelBase):
            if model.is_frozen and not save_frozen_weights:
                return
            if model_name_to_save is not None and name != model_name_to_save:
                return
            # save weights with ctor_kwargs
            if save_weights:
                model_uri = self._out_path / f"{name} cp={ckpt} model.th"
                torch.save(dict(state_dict=model.state_dict(), ctor_kwargs=model.ctor_kwargs), model_uri)
                self.logger.info(f"saved {name} to {model_uri}")
            elif save_latest_weights:
                # save only latest weights (and overwrite old latest weights)
                model_uri = self._out_path / f"{name} cp=latest model.th"
                torch.save(dict(state_dict=model.state_dict(), ctor_kwargs=model.ctor_kwargs), model_uri)
                self.logger.info(f"saved {name} to {model_uri}")
            # save optim
            if model.optim is not None:
                if save_optim:
                    optim_uri = self._out_path / f"{name} cp={ckpt} optim.th"
                elif save_latest_optim:
                    # save only latest optim (and overwrite old latest optim)
                    optim_uri = self._out_path / f"{name} cp=latest optim.th"
                else:
                    optim_uri = None
                if optim_uri is not None:
                    torch.save(model.optim.state_dict(), optim_uri)
                    self.logger.info(f"saved {name} optim to {optim_uri}")

            # save ctor kwargs
            if save_weights:
                kwargs_uri = self._out_path / f"{name} kwargs.yaml"
                if not kwargs_uri.exists():
                    with open(kwargs_uri, "w") as f:
                        yaml.safe_dump(model.ctor_kwargs, f)
        elif isinstance(model, CompositeModelBase):
            for k, v in model.submodels.items():
                self._save_seperate_models(
                    name=f"{name}.{k}",
                    model=v,
                    ckpt=ckpt,
                    save_weights=save_weights,
                    save_optim=save_optim,
                    save_latest_weights=save_latest_weights,
                    save_latest_optim=save_latest_optim,
                    model_name_to_save=model_name_to_save,
                    save_frozen_weights=save_frozen_weights,
                )
            # save ctor kwargs
            if model_name_to_save is not None:
                kwargs_uri = self._out_path / f"{name} kwargs.yaml"
                if not kwargs_uri.exists():
                    with open(kwargs_uri, "w") as f:
                        yaml.safe_dump(model.ctor_kwargs, f)
        else:
            raise NotImplementedError

    def _save_single(self, trainer_sd, model, ckpt, save_weights, save_optim):
        assert trainer_sd is not None
        sd = {}
        if save_weights:
            sd.update(
                trainer=trainer_sd,
                model=self._get_model_state_dict(model),
                model_ctor_kwargs=model.ctor_kwargs,
            )
        if save_optim:
            sd["optim"] = self._get_model_state_dict(model)
        ckpt_uri = self._out_path / f"{ckpt}.th"
        torch.save(sd, ckpt_uri)
        self.logger.info(f"saved checkpoint {ckpt} to {ckpt_uri}")

    @staticmethod
    def _get_model_state_dict(model):
        if isinstance(model, SingleModelBase):
            return CheckpointWriter._unwrapp_model(model).state_dict()
        elif isinstance(model, CompositeModelBase):
            state_dict = {}
            for k, v in model.submodels.items():
                state_dict[k] = CheckpointWriter._get_model_state_dict(v)
            return state_dict
        else:
            raise NotImplementedError

    @staticmethod
    def _get_optim_state_dict(model):
        if isinstance(model, SingleModelBase):
            assert model.optim is not None
            return CheckpointWriter._unwrapp_model(model).optim.state_dict()
        elif isinstance(model, CompositeModelBase):
            state_dict = {}
            for k, v in model.submodels.items():
                if v.optim is not None:
                    state_dict[k] = CheckpointWriter._get_optim_state_dict(v)
            return state_dict
        else:
            raise NotImplementedError
