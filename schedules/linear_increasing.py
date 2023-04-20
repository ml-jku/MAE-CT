from .base.increasing_schedule import IncreasingSchedule


class LinearIncreasing(IncreasingSchedule):
    def _get_value(self, step, total_steps):
        progress = step / max(1, total_steps - 1)
        return progress

    def __str__(self):
        return "LinearIncreasing"

    # LEGACY
    @property
    def is_new_schedule(self):
        return True
