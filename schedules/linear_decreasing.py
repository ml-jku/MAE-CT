from .base.decreasing_schedule import DecreasingSchedule


class LinearDecreasing(DecreasingSchedule):
    def _get_value(self, step, total_steps):
        # equivalent to the increasing schedule -> inversion is handled by IncreasingSchedule
        progress = step / max(1, total_steps - 1)
        return progress

    def __str__(self):
        return "LinearDecreasing"

    # LEGACY
    @property
    def is_new_schedule(self):
        return True
