class NoopContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass
