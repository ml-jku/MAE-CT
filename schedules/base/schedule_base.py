class ScheduleBase:
    def get_value(self, step, total_steps):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError
