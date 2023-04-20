import torch


class ConcatHalfFinalizer:
    """ concats input then takes the first half (e.g. when multiple views are stacked together for forward pass) """

    def __init__(self, index=0):
        assert index in [0, 1]
        self.index = index

    def __call__(self, features):
        features = torch.concat(features)
        if self.index == 0:
            return features[:len(features) // 2]
        return features[len(features) // 2:]

    def __str__(self):
        return f"{type(self).__name__}({self.index})"
