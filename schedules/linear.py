from .base.legacy_schedule import LegacySchedule


# LEGACY: old yamls use this
class Linear(LegacySchedule):
    # noinspection PyUnusedLocal
    def __init__(self, start_value=0., end_value=1., exclude_last=False, abs_max_value=None, **kwargs):
        super().__init__(**kwargs)
        self.start_value = start_value
        self.end_value = end_value
        assert 0. <= start_value < 1.
        assert 0. < end_value <= 1.
        assert start_value <= end_value
        self.delta = self.end_value - self.start_value
        self.exclude_last = exclude_last

    def _get_value(self, step, total_steps):
        progress = step / max(1, total_steps - 1)
        return self.start_value + self.delta * progress
