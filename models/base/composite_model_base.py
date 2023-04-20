from optimizers.composite_optimizer import CompositeOptimizer
from .model_base import ModelBase


class CompositeModelBase(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._optim = CompositeOptimizer(self)

    def forward(self, *args, **kwargs):
        """ all computations for training have to be within the forward method (otherwise DDP doesn't sync grads) """
        raise NotImplementedError

    @property
    def submodels(self):
        raise NotImplementedError

    @property
    def device(self):
        devices = [sub_model.device for sub_model in self.submodels.values()]
        assert all(device == devices[0] for device in devices[1:])
        return devices[0]

    @property
    def is_batch_size_dependent(self):
        return any(m.is_batch_size_dependent for m in self.submodels.values())

    def should_apply_model_specific_initialization(self):
        for m in self.submodels.values():
            if isinstance(m, CompositeModelBase):
                if not m.should_apply_model_specific_initialization():
                    return False
            else:
                if m.initializer is not None and not m.initializer.should_apply_model_specific_initialization:
                    return False
        return True

    def initialize_weights(self, config_provider=None, summary_provider=None):
        for sub_model in self.submodels.values():
            sub_model.initialize_weights(config_provider=config_provider, summary_provider=summary_provider)
        if self._model_specific_initialization != ModelBase._model_specific_initialization:
            if self.should_apply_model_specific_initialization():
                self.logger.info(f"applying model specific initialization")
                self._model_specific_initialization()
            else:
                self.logger.info(f"skipping model specific initialization")
        else:
            self.logger(f"no model specific initialization")
        return self

    def initialize_optim(self, lr_scaler_factor):
        for sub_model in self.submodels.values():
            sub_model.initialize_optim(lr_scaler_factor=lr_scaler_factor)

    def train(self, mode=True):
        for sub_model in self.submodels.values():
            sub_model.train(mode=mode)
        super().train(mode=mode)

    def to(self, device, *args, **kwargs):
        for sub_model in self.submodels.values():
            sub_model.to(*args, **kwargs, device=device)
        return self
