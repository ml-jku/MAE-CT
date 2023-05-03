import logging
from copy import deepcopy

import torch.nn as nn

from initializers import initializer_from_kwargs
from initializers.base.checkpoint_initializer import CheckpointInitializer
from utils.factory import instantiate, get_ctor


def act_ctor_from_kwargs(kind, **kwargs):
    return get_ctor(module_names=["torch.nn"], type_names=[kind], **kwargs)


def norm_ctor_from_kwargs(kind, **kwargs):
    if kind == "nonorm":
        kind = "identity"
    return get_ctor(module_names=["torch.nn"], type_names=[kind], **kwargs)


def model_from_kwargs(kind=None, stage_path_provider=None, **kwargs):
    # exclude update_counter from copying (otherwise model and trainer have different update_counter objects)
    update_counter = kwargs.pop("update_counter", None)
    ctx = kwargs.pop("ctx", None)
    kwargs = deepcopy(kwargs)

    # allow setting multiple kwargs in yaml; but allow also overwriting it
    # kind: vit.masked_encoder
    # kwargs: ${select:${vars.encoder_model_key}:${yaml:models/vit}}
    # patch_size: [128, 1] # this will overwrite the patch_size in kwargs
    kwargs_from_yaml = kwargs.pop("kwargs", {})
    kwargs = {**kwargs_from_yaml, **kwargs}

    # try to load kwargs from checkpoint
    if "initializer" in kwargs:
        initializer_kwargs = kwargs["initializer"]
        # TODO cleaner way to not load kwargs from checkpoint
        use_checkpoint_kwargs = initializer_kwargs.pop("use_checkpoint_kwargs", False)
        initializer = initializer_from_kwargs(**initializer_kwargs, stage_path_provider=stage_path_provider)
        if isinstance(initializer, CheckpointInitializer) and use_checkpoint_kwargs:
            ckpt_kwargs = initializer.get_kwargs()
            kind = ckpt_kwargs.pop("kind", None)
            # initializer is allowed to be different
            ckpt_kwargs.pop("initializer", None)
            ckpt_kwargs.pop("optim_ctor", None)
            # freezers shouldnt be used
            ckpt_kwargs.pop("freezers", None)
            ckpt_kwargs.pop("is_frozen", None)
            # check if keys overlap; this can be intended
            # - masked_encoder passes patch_size to decoder (in code)
            # - vit trained with drop_path_rate but then for evaluation this should be set to 0
            # if keys overlap the explicitly specified value dominates (i.e. from yaml or from code)
            kwargs_intersection = set(kwargs.keys()).intersection(set(ckpt_kwargs.keys()))
            if len(kwargs_intersection) > 0:
                logging.info(f"checkpoint_kwargs overlap with kwargs (intersection={kwargs_intersection})")
                for intersecting_kwarg in kwargs_intersection:
                    ckpt_kwargs.pop(intersecting_kwarg)
            kwargs.update(ckpt_kwargs)
    assert kind is not None, "model has no kind (maybe use_checkpoint_kwargs=True is missing in the initializer?)"

    # rename optim to optim_ctor (in yaml it is intuitive to call it optim as the yaml should not bother with the
    # implementation details but the implementation passes a ctor so it should also be called like it)
    optim = kwargs.pop("optim", None)

    # remove optim from model (e.g. for EMA)
    if optim is not None:
        kwargs["optim_ctor"] = optim

    # filter out modules passed to ctor (e.g. SslModel passes backbone to its heads)
    ctor_kwargs_filtered = {k: v for k, v in kwargs.items() if not isinstance(v, nn.Module)}
    ctor_kwargs = deepcopy(ctor_kwargs_filtered)
    ctor_kwargs["kind"] = kind
    ctor_kwargs.pop("input_shape", None)
    ctor_kwargs.pop("output_shape", None)

    return instantiate(
        module_names=[
            f"models.{kind}",
            f"models.composite.{kind}",
        ],
        type_names=[kind.split(".")[-1]],
        update_counter=update_counter,
        stage_path_provider=stage_path_provider,
        ctx=ctx,
        ctor_kwargs=ctor_kwargs,
        **kwargs,
    )


def remove_all_optims_from_kwargs(kwargs):
    # remove optim from all SingleModels (e.g. used for EMA)
    kwargs = deepcopy(kwargs)
    _remove_all_optims_from_kwargs(kwargs)
    return kwargs


def _remove_all_optims_from_kwargs(kwargs):
    if not isinstance(kwargs, dict):
        return
    kwargs.pop("optim", None)
    for v in kwargs.values():
        _remove_all_optims_from_kwargs(v)
