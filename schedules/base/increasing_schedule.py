from .basic_schedule import BasicSchedule


class IncreasingSchedule(BasicSchedule):
    def __init__(self, abs_start_value=0., abs_max_value=1., **kwargs):
        delta = abs_max_value - abs_start_value
        assert delta >= 0.
        super().__init__(abs_start_value=abs_start_value, abs_delta=delta, **kwargs)

    def _get_value(self, step, total_steps):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
