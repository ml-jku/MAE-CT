from copy import deepcopy
from functools import partial

from optimizers.optimizer_wrapper import OptimizerWrapper
from utils.factory import get_ctor


def optim_ctor_from_kwargs(kind, update_counter, **kwargs):
    kwargs = deepcopy(kwargs)

    # extract optimizer wrapper kwargs
    wrapped_optim_kwargs = {}
    wrapped_optim_kwargs_keys = [
        "schedule",
        "weight_decay_schedule",
        "clip_grad_value",
        "clip_grad_norm",
        "exclude_bias_from_wd",
        "exclude_norm_from_wd",
        "lr_scaler",
        "param_group_modifiers",
    ]
    for key in wrapped_optim_kwargs_keys:
        if key in kwargs:
            wrapped_optim_kwargs[key] = kwargs.pop(key)

    # all implementations use the default lr scaler with divisor=256 -> use this by default
    # this is added here instead of directly in the class to avoid changing the unittests that don't use a lr scaler
    # but it should actually be within the OptimizerWrapper class
    if "lr_scaler" not in wrapped_optim_kwargs:
        wrapped_optim_kwargs["lr_scaler"] = dict(kind="linear_lr_scaler")

    # add info for schedules
    wrapped_optim_kwargs["end_checkpoint"] = update_counter.end_checkpoint
    wrapped_optim_kwargs["updates_per_epoch"] = update_counter.updates_per_epoch
    wrapped_optim_kwargs["effective_batch_size"] = update_counter.effective_batch_size

    torch_optim_ctor = get_ctor(
        module_names=[f"torch.optim", f"optimizers.custom.{kind}"],
        type_names=[kind],
        **kwargs,
    )
    return partial(_optimizer_wrapper_ctor, torch_optim_ctor=torch_optim_ctor, **wrapped_optim_kwargs)


def _optimizer_wrapper_ctor(model, torch_optim_ctor, **wrapped_optim_kwargs):
    return OptimizerWrapper(model=model, torch_optim_ctor=torch_optim_ctor, **wrapped_optim_kwargs)
