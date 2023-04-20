class NoopGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        pass

    @staticmethod
    def step(optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    @staticmethod
    def update():
        pass
