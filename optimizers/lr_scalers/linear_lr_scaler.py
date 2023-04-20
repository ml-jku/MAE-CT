from .lr_scaler import LrScaler


class LinearLrScaler(LrScaler):
    def __init__(self, divisor=256, **kwargs):
        super().__init__(**kwargs)
        self.divisor = divisor

    def get_scaled_lr(self, lr_scaler_factor):
        return self.base_lr * lr_scaler_factor / self.divisor

    def __str__(self):
        return f"{type(self).__name__}(divisor={self.divisor})"
