class RunningMean:
    def __init__(self):
        self.count = None
        self.mean = None
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0

    def update(self, value, count=1):
        if value.ndim == 0:
            # count == 1: value to update
            # count >= 2: value is already an average of <count> values
            value_sum = value * count
        elif value.ndim == 1:
            count = len(value)
            value_sum = value.sum()
        else:
            raise NotImplementedError
        self.count += count

        # https://stackoverflow.com/a/23493727/13253318
        self.mean = self.mean * (self.count - count) / self.count + value_sum / self.count
