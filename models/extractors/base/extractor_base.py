import torch
import torch.nn as nn

from models.extractors.finalizers import finalizer_from_kwargs
from models.extractors.finalizers.concat_finalizer import ConcatFinalizer
from models.poolings.single_pooling import SinglePooling
from utils.factory import create
from utils.select_with_path import select_with_path


class ExtractorBase:
    def __init__(self, pooling=None, finalizer=ConcatFinalizer, model_path=None, outputs=None):
        self.pooling_kwargs = pooling
        self.pooling = None
        # "self.outputs = outputs or {}" does not work here as an empty dictionary evaluates to false
        if outputs is None:
            self.outputs = {}
        else:
            self.outputs = outputs
        self.hooks = []
        self.finalizer = create(finalizer, finalizer_from_kwargs)
        self.model_path = model_path
        self.registered_hooks = False

    @staticmethod
    def _concat(features):
        return torch.concat(features, dim=1)

    @staticmethod
    def _mean(features):
        return torch.stack(features).mean(dim=0)

    def __repr__(self):
        return str(self)

    def __str__(self):
        model_path = f"{self.model_path}." if self.model_path is not None else ""
        finalize_str = f".{str(self.finalizer)}" if not isinstance(self.finalizer, ConcatFinalizer) else ""
        return f"{model_path}{self.to_string()}{finalize_str}"

    def to_string(self):
        raise NotImplementedError

    def register_hooks(self, model):
        assert not self.registered_hooks
        model = select_with_path(obj=model, path=self.model_path)
        self.pooling = create(self.pooling_kwargs, SinglePooling) or nn.Identity()
        self._register_hooks(model)
        self.registered_hooks = True

    def _register_hooks(self, model):
        raise NotImplementedError

    def enable_hooks(self):
        for hook in self.hooks:
            hook.enabled = True

    def disable_hooks(self):
        for hook in self.hooks:
            hook.enabled = False

    def extract(self):
        assert len(self.outputs) > 0
        features = [self.pooling(output) for output in self.outputs.values()]
        if self.finalizer is not None:
            features = self.finalizer(features)
        self.outputs.clear()
        return features

    # def features(self, model, *args, **kwargs):
    #     assert len(self.outputs) == 0, "clear Extractor.outputs when not needed anymore to avoid memory leaks"
    #     self.enable_raise_exception()
    #     try:
    #         _ = model.features(*args, **kwargs)
    #     except StopForwardException:
    #         pass
    #     # return a copy and clear for new forward pass
    #     outputs = {k: v for k, v in self.outputs.items()}
    #     self.outputs.clear()
    #     return outputs
