from itertools import product

import torch.nn as nn

from models.base.composite_model_base import CompositeModelBase
from .linear_head import LinearHead
from .simclrv2_linear_head import Simclrv2LinearHead


class MultiLinearHead(CompositeModelBase):
    def __init__(
            self,
            head_kind="heads.linear_head",
            head_kwargs=None,
            poolings=None,
            optimizers=None,
            initializers=None,
            nonaffine_batchnorm=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        if nonaffine_batchnorm is None:
            bn_search_space = [True]
        elif isinstance(nonaffine_batchnorm, list):
            bn_search_space = nonaffine_batchnorm
        else:
            bn_search_space = [nonaffine_batchnorm]
        search_space = product(
            poolings.items() if poolings is not None else [(None, None)],
            optimizers.items() if optimizers is not None else [(None, None)],
            initializers.items() if initializers is not None else [(None, None)],
            bn_search_space,
        )
        layers = {}
        for (pool_name, pooling), (optim_name, optimizer), (init_name, initializer), use_bn in search_space:
            ctor_kwargs = dict(
                pooling=pooling,
                optim_ctor=optimizer,
                initializer=initializer,
                nonaffine_batchnorm=use_bn,
            )
            if head_kind == "heads.linear_head":
                head_cls = LinearHead
            elif head_kind == "heads.simclrv2_linear_head":
                head_cls = Simclrv2LinearHead
            else:
                raise NotImplementedError
            layer = head_cls(
                **ctor_kwargs,
                **head_kwargs or {},
                input_shape=self.input_shape,
                output_shape=self.output_shape,
                update_counter=self.update_counter,
                stage_path_provider=self.stage_path_provider,
                ctor_kwargs=dict(**ctor_kwargs, **head_kwargs or {}, kind=head_kind),
                ctx=self.ctx,
            )
            names = []
            if pooling is not None:
                names.append(pool_name)
            if optimizer is not None:
                names.append(optim_name)
            if initializers is not None:
                names.append(init_name)
            if len(bn_search_space) > 1:
                if use_bn:
                    names.append("bn")
                else:
                    names.append("nobn")
            name = "_".join(names)

            layers[name] = layer
        self.layers = nn.ModuleDict(layers)

    @property
    def submodels(self):
        return self.layers

    def forward(self, x):
        return {key: layer(x) for key, layer in self.layers.items()}

    def features(self, x):
        return self(x)

    def predict(self, x):
        result = {}
        for key, layer in self.layers.items():
            result[key] = layer.predict(x)["main"]
        return result
