from copy import deepcopy

from utils.factory import instantiate


def schedule_from_kwargs(schedule_configs, update_counter, **kwargs):
    if schedule_configs is None:
        return None
    return schedule_from_schedule_configs(
        schedule_configs,
        end_checkpoint=update_counter.end_checkpoint,
        effective_batch_size=update_counter.effective_batch_size,
        updates_per_epoch=update_counter.updates_per_epoch,
        **kwargs,
    )


def schedule_from_schedule_configs(schedule_configs, **kwargs):
    if schedule_configs is None:
        return None
    # only allow lists for unambiguous definitions (e.g. single schedules could be defined with dict or with 1 element
    # list, but this would result in wandb config being different for two identical schedules)
    assert isinstance(schedule_configs, list) and all(isinstance(sc, dict) for sc in schedule_configs), \
        "a schedule schould be defined as a list of subschedules (also if only a single schedule is used)"
    schedule_configs = deepcopy(schedule_configs)
    # import here to avoid circular imports
    from .base.sequential_schedule import SequentialSchedule
    return SequentialSchedule(schedule_configs, **kwargs)


def basic_schedule_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"schedules.{kind}"], type_names=[kind], **kwargs)


def schedule_config_from_kwargs(**kwargs):
    schedule_kwargs = deepcopy(kwargs)
    config_kwargs = {}
    for kwarg_key in ["mode", "start_percent", "end_percent", "start_checkpoint", "end_checkpoint"]:
        if kwarg_key in schedule_kwargs:
            config_kwargs[kwarg_key] = schedule_kwargs.pop(kwarg_key)

    assert "kind" in schedule_kwargs
    from .base.schedule_config import ScheduleConfig
    return ScheduleConfig(schedule_kwargs, **config_kwargs)
