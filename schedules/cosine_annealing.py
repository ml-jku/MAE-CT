import math

from .base.legacy_schedule import LegacySchedule


# LEGACY: old yamls use this
class CosineAnnealing(LegacySchedule):
    # noinspection PyUnusedLocal
    def __init__(self, abs_max_value=None, **kwargs):
        super().__init__(**kwargs)

    def _get_value(self, step, total_steps):
        progress = step / max(1, total_steps - 1)
        return (1 + math.cos(math.pi * progress)) / 2
