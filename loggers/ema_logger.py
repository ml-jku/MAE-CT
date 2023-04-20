from collections import defaultdict

import torch

from distributed.config import is_rank0
from loggers.base.logger_base import LoggerBase
from utils.select_with_path import select_with_path


class EmaLogger(LoggerBase):
    def __init__(self, target_factors, model_paths=None, **kwargs):
        super().__init__(**kwargs)
        self.model_paths = model_paths or [None]
        self.target_factors = target_factors
        self.parameters = defaultdict(dict)
        self.buffers = defaultdict(dict)

    def _before_training(self, model, **kwargs):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                for name, param in cur_model.named_parameters():
                    self.parameters[(model_path, target_factor)][name] = param.clone()
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name] = buffer.clone()

    def _track_after_update_step(self, model, **kwargs):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            cur_model = select_with_path(obj=model, path=model_path)
            for target_factor in self.target_factors:
                for name, param in cur_model.named_parameters():
                    key = (model_path, target_factor)
                    self.parameters[key][name].data.mul_(target_factor).add_(param.data * (1. - target_factor))
            for name, buffer in cur_model.named_buffers():
                self.buffers[model_path][name].data.copy_(buffer.data)

    def _save(self, ckpt, model):
        if not is_rank0():
            return
        for model_path in self.model_paths:
            for target_factor in self.target_factors:
                state_dict = {**self.parameters[(model_path, target_factor)], **self.buffers[model_path]}
                if model_path is None:
                    cur_model_path = model.name
                else:
                    cur_model_path = f"{model.name}.{model_path}"
                fname = f"{cur_model_path} cp={ckpt} ema={target_factor} model.th"
                torch.save(state_dict, self.stage_path_provider.checkpoint_path / fname)

    def _log(self, update_counter, model, **kwargs):
        self._save(ckpt=update_counter.cur_checkpoint, model=model)

    def _after_training(self, model, **kwargs):
        self._save(ckpt="last", model=model)
