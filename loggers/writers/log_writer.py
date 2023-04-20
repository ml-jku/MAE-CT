import logging
from collections import defaultdict

import torch
import wandb
import yaml

from providers.stage_path_provider import StagePathProvider


class LogWriter:
    def __init__(self, stage_path_provider: StagePathProvider):
        self.logger = logging.getLogger(type(self).__name__)
        self.stage_path_provider = stage_path_provider
        self.log_entries = []
        self.log_cache = None
        self.is_wandb = wandb.run is not None

    def finish(self):
        if len(self.log_entries) == 0:
            return
        entries_uri = self.stage_path_provider.primitive_entries_uri
        self.logger.info(f"writing {len(self.log_entries)} log entries to {entries_uri}")
        # convert into {<key>: {<update0>: <value0>, <update1>: <value1>}}
        result = defaultdict(dict)
        for entry in self.log_entries:
            # update is used instead of wandb's _step
            update = entry["update"]
            for key, value in entry.items():
                if key == "update":
                    continue
                result[key][update] = value
        with open(entries_uri, "w") as f:
            yaml.safe_dump(dict(result), f)

    def _log(self, key, value, update_counter):
        if self.log_cache is None:
            self.log_cache = dict(
                epoch=update_counter.epoch,
                update=update_counter.update,
                sample=update_counter.sample,
            )
        self.log_cache[key] = value

    def flush(self):
        if self.log_cache is None:
            return
        if self.is_wandb:
            wandb.log(self.log_cache)
        # wandb doesn't support querying offline logfiles so offline mode would have no way to summarize stages
        # also fetching the summaries from the online version potentially takes a long time, occupying GPU servers
        # for primitive tasks
        # -------------------
        # wandb also has weird behavior when lots of logs are done seperately -> collect all log values and log once
        # -------------------
        # check that every log is fully cached (i.e. no update is logged twice)
        if len(self.log_entries) > 0:
            assert self.log_cache["update"] > self.log_entries[-1]["update"]
        # don't keep histograms for primitive logging
        self.log_entries.append({k: v for k, v in self.log_cache.items() if not isinstance(v, wandb.Histogram)})
        self.log_cache = None

    def add_scalar(self, key, value, update_counter):
        if torch.is_tensor(value):
            value = value.item()
        self._log(key, value, update_counter)

    def add_histogram(self, key, data, update_counter):
        if self.is_wandb:
            self._log(key, wandb.Histogram(data), update_counter)
