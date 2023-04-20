import logging


class CompositeOptimizer:
    def __init__(self, composite_model):
        self.logger = logging.getLogger(type(self).__name__)
        self.composite_model = composite_model

    def step(self, grad_scaler):
        for model in self.composite_model.submodels.values():
            if model.optim is None:
                continue
            # self.logger.info(f"{model.name} optim.step")
            model.optim.step(grad_scaler)
        self.composite_model.after_update_step()

    def schedule_epoch_step(self, epoch):
        for model in self.composite_model.submodels.values():
            if model.optim is None:
                continue
            model.optim.schedule_epoch_step(epoch)

    def schedule_update_step(self, checkpoint):
        for model in self.composite_model.submodels.values():
            if model.optim is None:
                continue
            model.optim.schedule_update_step(checkpoint)

    def zero_grad(self, set_to_none=True):
        for model in self.composite_model.submodels.values():
            if model.optim is None:
                continue
            model.optim.zero_grad(set_to_none)
