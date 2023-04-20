from .basic_schedule import BasicSchedule


# LEGACY: old schedules were increasing by default
class LegacySchedule(BasicSchedule):
    def __init__(self):
        super().__init__(abs_start_value=0., abs_delta=1.)

    def _get_value(self, step, total_steps):
        raise NotImplementedError

    def __str__(self):
        return "LegacySchedule"
