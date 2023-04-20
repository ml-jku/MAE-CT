import math

from .base.decreasing_schedule import DecreasingSchedule


class CosineDecreasing(DecreasingSchedule):
    def _get_value(self, step, total_steps):
        # this is returns progress values (i.e. goes from 0 to 1)
        # the progress is then adapted to be decreasing by DecreasingSchedule
        progress = step / max(1, total_steps - 1)
        return 1 - (1 + math.cos(math.pi * progress)) / 2

    def __str__(self):
        return "CosineDecreasing"

    # LEGACY
    @property
    def is_new_schedule(self):
        return True
