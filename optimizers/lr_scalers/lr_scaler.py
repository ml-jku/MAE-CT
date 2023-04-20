class LrScaler:
    def __init__(self, base_lr):
        super().__init__()
        self.base_lr = base_lr

    def get_scaled_lr(self, lr_scaler_factor):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError
