from .base.schedule_base import ScheduleBase


class Constant(ScheduleBase):
    # TODO abs_max_value should not be passed into here
    # noinspection PyUnusedLocal
    def __init__(self, value, abs_max_value=None):
        super().__init__()
        self.value = value

    def __str__(self):
        return "Constant"

    def get_value(self, step, total_steps):
        assert 0 <= step < total_steps, f"0 <= step < total_steps (step={step} total_steps={total_steps})"
        return self.value

    # LEGACY
    @property
    def is_new_schedule(self):
        return False
