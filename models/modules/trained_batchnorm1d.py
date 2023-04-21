import torch.nn as nn


class TrainedBatchNorm1d(nn.BatchNorm1d):
    """ lazy solution to use the statistics of a trained batchnorm """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train(mode=False)

    def train(self, mode: bool = True):
        return super().train(mode=False)
