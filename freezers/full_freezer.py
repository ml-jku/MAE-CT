from .base.freezer_base import FreezerBase


class FullFreezer(FreezerBase):
    def __str__(self):
        return type(self).__name__

    def _change_state(self, model, requires_grad):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _before_accumulation_step(self, model):
        model.eval()
