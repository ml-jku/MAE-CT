import logging

import torch.nn

from providers.stage_path_provider import StagePathProvider
from utils.naming_util import snake_type_name


class ModelBase(torch.nn.Module):
    def __init__(
            self,
            input_shape=None,
            name=None,
            output_shape=None,
            ctor_kwargs=None,
            update_counter=None,
            stage_path_provider: StagePathProvider = None,
            ctx: dict = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.update_counter = update_counter
        self.stage_path_provider = stage_path_provider
        self._optim = None
        # a context allows extractors to store activations for later pooling (e.g. use features from last 4 layers)
        # the context has to be cleared manually after every call (e.g. model.features) to avoid memory leaks
        # "self.outputs = outputs or {}" does not work here as an empty dictionary evaluates to false
        if ctx is None:
            self.ctx = {}
        else:
            self.ctx = ctx

        # allow setting name of model manually (useful if a standalone model is trained in multiple stages
        # then the checkpoint from the previous stage is only the name; if the typename is used for this,
        # the checkpoint loader would have to be changed when the model type changes; if the name is set for this case
        # it doesn't have to be changed)
        self.name = name or snake_type_name(self)
        # store the kwargs that are relevant
        self.ctor_kwargs = ctor_kwargs
        # don't save update_counter in save_kwargs
        if self.ctor_kwargs is not None and "update_counter" in self.ctor_kwargs:
            self.ctor_kwargs.pop("update_counter")
        # flag to make sure the model was initialized before wrapping into DDP
        # (parameters/buffers are synced in __init__ of DDP, so if model is not initialized before that,
        # different ranks will have diffefent parameters because the seed is different for every rank)
        # different seeds per rank are needed to avoid stochastic processes being the same across devices
        # (e.g. if seeds are equal, all masks for MAE are the same per batch)
        self.is_initialized = False

    def forward(self, *args, **kwargs):
        """ all computations for training have to be within the forward method (otherwise DDP doesn't sync grads) """
        raise NotImplementedError

    @property
    def submodels(self):
        raise NotImplementedError

    @property
    def unwrapped_ddp_module(self):
        return self

    @property
    def is_batch_size_dependent(self):
        raise NotImplementedError

    def initialize(self, lr_scaler_factor=None, config_provider=None, summary_provider=None):
        self.initialize_weights(config_provider=config_provider, summary_provider=summary_provider)
        self.initialize_optim(lr_scaler_factor=lr_scaler_factor)
        self.is_initialized = True
        return self

    def initialize_weights(self, config_provider=None, summary_provider=None):
        raise NotImplementedError

    def initialize_optim(self, lr_scaler_factor):
        raise NotImplementedError

    def _model_specific_initialization(self):
        pass

    @property
    def optim(self):
        return self._optim

    @property
    def device(self):
        raise NotImplementedError

    def before_accumulation_step(self):
        """ before_accumulation_step hook (e.g. for freezers) """
        for model in self.submodels.values():
            model.before_accumulation_step()

    def after_update_step(self):
        """ after_update_step hook (e.g. for EMA) """
        pass
