from .schedule_base import ScheduleBase


# TODO rename to something like numeric schedule (and also in the objects that contain basic schedules)
class BasicSchedule(ScheduleBase):
    def __init__(self, abs_start_value, abs_delta, exclude_first=False, exclude_last=False):
        self.abs_start_value = abs_start_value
        self.abs_delta = abs_delta
        self.exclude_first = exclude_first
        self.exclude_last = exclude_last

    def __str__(self):
        raise NotImplementedError

    # LEGACY
    @property
    def is_new_schedule(self):
        return False

    def get_value(self, step, total_steps):
        assert 0 <= step < total_steps, f"0 <= step < total_steps (step={step} total_steps={total_steps})"
        if self.exclude_last:
            total_steps += 1
        if self.exclude_first:
            step += 1
            total_steps += 1
        # get value from schedule (in [0, 1])
        value = self._get_value(step, total_steps)
        # adjust to "absolute value" (i.e. real learning rate)
        value = self.abs_start_value + value * self.abs_delta
        return value

    def _get_value(self, step, total_steps):
        raise NotImplementedError
