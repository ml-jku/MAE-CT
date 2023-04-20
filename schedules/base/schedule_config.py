from schedules import basic_schedule_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create


class ScheduleConfig:
    def __init__(
            self,
            schedule,
            mode="before_update",
            start_percent=None,
            end_percent=None,
            start_checkpoint=None,
            end_checkpoint=None,
            **kwargs,
    ):
        self.basic_schedule = create(schedule, basic_schedule_from_kwargs, **kwargs)
        self.mode = mode
        assert self.mode in ["before_epoch", "before_update"]
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.start_checkpoint = create(start_checkpoint, Checkpoint)
        self.end_checkpoint = create(end_checkpoint, Checkpoint)

    @property
    def step_before_update(self):
        return self.mode == "before_update"

    @property
    def step_before_epoch(self):
        return self.mode == "before_epoch"

    @property
    def total_steps(self):
        if self.step_before_update:
            return self.end_checkpoint.update - self.start_checkpoint.update
        if self.step_before_epoch:
            return self.end_checkpoint.epoch - self.start_checkpoint.epoch
        raise NotImplementedError

    def get_current_step(self, checkpoint):
        assert self.start_checkpoint <= checkpoint <= self.end_checkpoint
        if self.step_before_epoch:
            return checkpoint.epoch - self.start_checkpoint.epoch
        if self.step_before_update:
            return checkpoint.update - self.start_checkpoint.update
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.basic_schedule)
