from torch.utils.data.sampler import BatchSampler


class InfiniteBatchSampler(BatchSampler):
    def __iter__(self):
        while True:
            yield from super().__iter__()

    def __len__(self):
        raise NotImplementedError
