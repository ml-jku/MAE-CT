class NoopWriter:
    def __init__(self, *_, **__):
        pass

    def __getattr__(self, item):
        return self._do_nothing

    def _do_nothing(self, *_, **__):
        pass

    def __setattr__(self, key, value):
        pass
