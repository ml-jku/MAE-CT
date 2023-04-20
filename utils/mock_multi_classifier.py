import numpy as np


class MultiClassifier:
    """ cyanure only works in linux  """

    def __init__(self, *_, **__):
        pass

    @staticmethod
    def fit(X, y, *_, **__):
        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float32
        assert isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype == np.int64

    @staticmethod
    def score(X, y, *_, **__):
        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float32
        assert isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype == np.int64
        return np.float64(0.1)

# https://github.com/inria-thoth/cyanure/blob/master/cyanure/data_processing.py#L21
# noinspection PyUnusedLocal
def preprocess(X, centering=False, normalize=True, columns=False):
    return X