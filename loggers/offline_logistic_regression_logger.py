import os
from collections import defaultdict

import torch
import torch.nn as nn

from distributed.config import is_rank0, barrier
from distributed.gather import all_gather_nograd_clipped
from models.poolings.single_pooling import SinglePooling
from utils.factory import create_collection
from utils.stdout_capturer import StdoutCapturer
from .base.logger_base import LoggerBase
from utils.subset_identifier import get_subset_identifier

class OfflineLogisticRegressionLogger(LoggerBase):
    def __init__(self, train_dataset_key, test_dataset_key, features_name, stage_id, **kwargs):
        super().__init__(**kwargs)
        self.train_dataset_key = train_dataset_key
        self.test_dataset_key = test_dataset_key
        self.features_name = features_name
        self.features_folder = self.stage_path_provider.output_path / "features" / stage_id / "features"

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, **_):
        if not is_rank0():
            return

        # setup logistic regression
        try:
            from cyanure import MultiClassifier, preprocess
        except ImportError:
            # cyanure is only available for linux -> mock on windows for development purposes
            assert os.name == "nt"
            from utils.mock_multi_classifier import MultiClassifier, preprocess

        # load features
        features_fname = f"%s-{self.features_name}-{update_counter.cur_checkpoint}-features.th"
        train_features_fname = features_fname % self.train_dataset_key
        test_features_fname = features_fname % self.test_dataset_key
        train_features = torch.load(self.features_folder / train_features_fname)
        test_features = torch.load(self.features_folder / test_features_fname)
        # load labels
        labels_fname = f"%s-labels.th"
        train_labels_fname = labels_fname % self.train_dataset_key
        test_labels_fname = labels_fname % self.test_dataset_key
        train_labels = torch.load(self.features_folder / train_labels_fname)
        test_labels = torch.load(self.features_folder / test_labels_fname)

        # sanity check
        assert len(train_features) == len(train_labels)
        assert len(test_features) == len(test_labels)
        assert train_features.ndim == 2
        assert test_features.ndim == 2

        # cyanure expects numpy arrays
        train_features = train_features.numpy()
        test_features = test_features.numpy()
        train_labels = train_labels.numpy()
        test_labels = test_labels.numpy()

        # fit
        self.logger.info(f"fit logistic regression of {len(train_features)} samples from '{train_features_fname}'")
        # https://github.com/facebookresearch/msn/blob/main/logistic_eval.py#L172
        preprocess(train_features, normalize=True, columns=False, centering=True)
        classifier = MultiClassifier(loss='multiclass-logistic', penalty="l2", fit_intercept=False)
        lamb = 0.00025 / len(train_features)
        with StdoutCapturer():
            classifier.fit(
                train_features,
                train_labels,
                it0=10,
                lambd=lamb,
                lambd2=lamb,
                nthreads=10,
                tol=1e-3,
                solver='auto',
                seed=0,
                max_epochs=300,
            )

        # predict
        self.logger.info(f"predict logistic regression")
        accuracy = float(classifier.score(test_features, test_labels))
        key = f"accuracy1/{self.train_dataset_key}/{self.features_name}"
        self.logger.info(f"{key}: {accuracy:.4f}")
        self.writer.add_scalar(key, accuracy, update_counter=update_counter)
