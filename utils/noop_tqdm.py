class NoopTqdm:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        pass

    def noop(self, *_, **__):
        pass

    def __getattr__(self, item):
        return self.noop
