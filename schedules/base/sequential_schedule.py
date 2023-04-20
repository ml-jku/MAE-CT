from typing import List

from schedules import schedule_config_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create_collection
from .schedule_config import ScheduleConfig


class SequentialSchedule:
    def __init__(
            self,
            schedule_configs: List[ScheduleConfig],
            end_checkpoint: Checkpoint = None,
            effective_batch_size: int = None,
            updates_per_epoch: int = None,
            **kwargs,
    ):
        assert isinstance(schedule_configs, list) and len(schedule_configs) > 0
        self.schedule_configs = create_collection(schedule_configs, schedule_config_from_kwargs, **kwargs)
        assert all(isinstance(sc, ScheduleConfig) for sc in self.schedule_configs)
        if end_checkpoint is not None and not end_checkpoint.is_fully_specified:
            end_checkpoint = end_checkpoint.to_fully_specified(
                updates_per_epoch=updates_per_epoch,
                effective_batch_size=effective_batch_size,
            )

        # initialize start/end checkpoints
        cur_start_checkpoint = Checkpoint(0, 0, 0)
        for i, schedule_config in enumerate(self.schedule_configs):
            # initialize start_checkpoint
            if schedule_config.start_checkpoint is None:
                if schedule_config.start_percent is not None:
                    assert end_checkpoint is not None
                    # percent based start_checkpoint
                    schedule_config.start_checkpoint = end_checkpoint.scale(
                        factor=schedule_config.start_percent / 100,
                        effective_batch_size=effective_batch_size,
                        updates_per_epoch=updates_per_epoch,
                        floor=True,
                    )
                else:
                    schedule_config.start_checkpoint = cur_start_checkpoint
            # initialize end_checkpoint
            if schedule_config.end_checkpoint is None:
                if schedule_config.end_percent is not None:
                    assert end_checkpoint is not None
                    # percent based end_checkpoint
                    schedule_config.end_checkpoint = end_checkpoint.scale(
                        factor=schedule_config.end_percent / 100,
                        effective_batch_size=effective_batch_size,
                        updates_per_epoch=updates_per_epoch,
                        floor=True,
                    )
                else:
                    assert i == len(
                        self.schedule_configs) - 1, "only last schedule_config can have implicit end_checkpoint"
                    schedule_config.end_checkpoint = end_checkpoint

            # fully specify checkpoints
            schedule_config.start_checkpoint = schedule_config.start_checkpoint.to_fully_specified(
                updates_per_epoch=updates_per_epoch,
                effective_batch_size=effective_batch_size,
            )
            schedule_config.end_checkpoint = schedule_config.end_checkpoint.to_fully_specified(
                updates_per_epoch=updates_per_epoch,
                effective_batch_size=effective_batch_size,
            )
            schedule_config.delta_ckpt = schedule_config.end_checkpoint - schedule_config.start_checkpoint

            # sanity checks
            assert schedule_config.start_checkpoint <= schedule_config.end_checkpoint, \
                "expecting start_checkpoint < end_checkpoint"
            if end_checkpoint is not None:
                assert schedule_config.end_checkpoint <= end_checkpoint

            cur_start_checkpoint = schedule_config.end_checkpoint

    def get_schedule_config(self, checkpoint):
        # return fitting schedule or schedule before checkpoint
        cur = self.schedule_configs[0]
        for cfg in self.schedule_configs[1:]:
            if cfg.start_checkpoint <= checkpoint:
                cur = cfg
            else:
                break
        return cur

    def should_step_before_epoch(self, checkpoint):
        if self.schedule_configs[0].start_checkpoint <= checkpoint <= self.schedule_configs[-1].end_checkpoint:
            return self.get_schedule_config(checkpoint).step_before_epoch
        return False

    def should_step_before_update(self, checkpoint):
        if self.schedule_configs[0].start_checkpoint <= checkpoint <= self.schedule_configs[-1].end_checkpoint:
            return self.get_schedule_config(checkpoint).step_before_update
        return False

    # LEGACY
    def get_is_new_schedule(self, checkpoint):
        schedule_config = self.get_schedule_config(checkpoint)
        return schedule_config.basic_schedule.is_new_schedule

    def get_value(self, checkpoint):
        schedule_config = self.get_schedule_config(checkpoint)
        if schedule_config.start_checkpoint <= checkpoint < schedule_config.end_checkpoint:
            step = schedule_config.get_current_step(checkpoint)
        else:
            if checkpoint < self.schedule_configs[0].start_checkpoint:
                # if checkpoint is before first schedule -> return first value of first schedule
                # (get_schedule_config returns first schedule)
                step = 0
            else:
                # gaps are allowed in a schedule -> return last value of previous schedule
                # (get_schedule_config returns last schedule)
                step = schedule_config.total_steps - 1
        return schedule_config.basic_schedule.get_value(step, schedule_config.total_steps)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "-".join(map(str, self.schedule_configs))
