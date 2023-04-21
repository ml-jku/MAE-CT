import torch

from freezers import freezer_from_kwargs
from initializers import initializer_from_kwargs
from initializers.functional import ALL_BATCHNORMS
from models.extractors import extractor_from_kwargs
from optimizers import optim_ctor_from_kwargs
from utils.factory import create, create_collection
from utils.model_utils import get_trainable_param_count
from .model_base import ModelBase


class SingleModelBase(ModelBase):
    def __init__(
            self,
            optim_ctor=None,
            freezers=None,
            initializer=None,
            is_frozen=False,
            update_counter=None,
            extractors=None,
            **kwargs
    ):
        super().__init__(update_counter=update_counter, **kwargs)
        self._device = torch.device("cpu")
        self.optim_ctor = create(
            optim_ctor,
            optim_ctor_from_kwargs,
            instantiate_if_ctor=False,
            update_counter=update_counter,
        )
        self.freezers = create_collection(freezers, freezer_from_kwargs, update_counter=update_counter)
        self.initializer = create(initializer, initializer_from_kwargs, stage_path_provider=self.stage_path_provider)
        self.extractors = create_collection(extractors, extractor_from_kwargs, outputs=self.ctx)
        self.is_frozen = is_frozen
        self._is_batch_size_dependent = None

        # check base methods were not overwritten
        assert type(self).before_accumulation_step == SingleModelBase.before_accumulation_step

    @property
    def is_batch_size_dependent(self):
        if self._is_batch_size_dependent is None:
            for m in self.modules():
                if isinstance(m, ALL_BATCHNORMS):
                    self._is_batch_size_dependent = True
                    break
            else:
                self._is_batch_size_dependent = False
        return self._is_batch_size_dependent

    def forward(self, *args, **kwargs):
        """ all computations for training have to be within the forward method (otherwise DDP doesn't sync grads) """
        raise NotImplementedError

    @property
    def submodels(self):
        return {self.name: self}

    @property
    def device(self):
        return self._device

    def before_accumulation_step(self):
        for freezer in self.freezers:
            freezer.before_accumulation_step(self)

    @property
    def _requires_initializer(self):
        """ pretrained/parameterless models don't need initializer """
        return True

    @property
    def should_apply_model_specific_initialization(self):
        return self.initializer is None or self.initializer.should_apply_model_specific_initialization

    def register_extractor_hooks(self):
        if len(self.extractors) > 0:
            for extractor in self.extractors:
                extractor.register_hooks(self)
                extractor.enable_hooks()
        return self

    def initialize_weights(self, config_provider=None, summary_provider=None):
        if self.initializer is None:
            if self._requires_initializer:
                self.logger.warning(f"{self.name}: no initializer (using torch default)")
        else:
            if not self._requires_initializer:
                self.initializer.init_weights(
                    model=self,
                    config_provider=config_provider,
                    summary_provider=summary_provider,
                )
            else:
                # initialize weights
                self.initializer.init_weights(
                    model=self,
                    config_provider=config_provider,
                    summary_provider=summary_provider,
                )

        # model specific initialization (e.g. GLOW initially sets last layer of coupling layer to 0)
        if self._model_specific_initialization != ModelBase._model_specific_initialization:
            # avoid overwriting checkpoint/pretrain initializations with model specific initialization
            if self.should_apply_model_specific_initialization:
                self.logger.info(f"{self.name} applying model specific initialization")
                self._model_specific_initialization()
            else:
                self.logger.info(f"{self.name} skipping model specific initialization")
        else:
            self.logger(f"{self.name} no model specific initialization")

        # freeze all parameters (and put into eval mode)
        if self.is_frozen:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

        # freeze some parameters
        for freezer in self.freezers:
            freezer.after_weight_init(self)

        return self

    def initialize_optim(self, lr_scaler_factor):
        if self.optim_ctor is not None:
            self.logger.info(f"{self.name} initialize optimizer")
            self._optim = self.optim_ctor(self, lr_scaler_factor=lr_scaler_factor)
            if self.initializer is not None:
                self.initializer.init_optim(self)
        elif not self.is_frozen:
            if get_trainable_param_count(self) == 0:
                self.logger.info(f"{self.name} has no trainable parameters -> freeze and put into eval mode")
                self.is_frozen = True
                self.eval()
            else:
                raise RuntimeError(f"no optimizer for {self.name} and it's also not frozen")
        else:
            self.logger.info(f"{self.name} is frozen -> no optimizer to initialize")

    def train(self, mode=True):
        # avoid setting mode to train if whole network is frozen
        # this prevents the training behavior of e.g. the following components
        # - Dropout/StochasticDepth dropping during
        # - BatchNorm (in train mode the statistics are tracked)
        if self.is_frozen and mode is True:
            return
        return super().train(mode=mode)

    def to(self, device, *args, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self._device = device
        return super().to(*args, **kwargs, device=device)
