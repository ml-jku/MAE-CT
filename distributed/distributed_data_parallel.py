from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel


class DistributedDataParallel(TorchDistributedDataParallel):
    """ wrapper that exposes all methods/fields of the wrapped module """

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)

    @property
    def unwrapped_ddp_module(self):
        return self.module
