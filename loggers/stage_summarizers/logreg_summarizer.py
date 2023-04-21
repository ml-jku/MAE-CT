from collections import defaultdict

import numpy as np

from .base.stage_summarizer_base import StageSummarizerBase


class LogregSummarizer(StageSummarizerBase):
    def summarize(self):
        # group values
        grouped = defaultdict(list)
        for key, value in self.all_log_entries.items():
            if "accuracy1" not in key:
                continue
            assert len(value) == 1
            key = key.replace("_split1", "").replace("_split2", "").replace("_split3", "")
            grouped[key].append(value[0])
        # stats
        for key, values in grouped.items():
            mean = float(np.mean(values))
            std = float(np.std(values))
            self.logger.info(f"{key}: mean={mean:.4f} std={std:.4f}")
            self.summary_provider[f"{key}/mean"] = mean
            self.summary_provider[f"{key}/std"] = std
