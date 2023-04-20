import numpy as np

from distributed.distributed_data_parallel import DistributedDataParallel
from loggers.base.summary_logger import SummaryLogger
from models.base.composite_model_base import CompositeModelBase
from utils.model_utils import get_trainable_param_count, get_frozen_param_count
from utils.naming_util import join_names, snake_type_name


class ParamCountLogger(SummaryLogger):
    @property
    def allows_no_interval_types(self):
        return True

    @staticmethod
    def _get_param_counts(model, trace=None):
        if isinstance(model, DistributedDataParallel):
            model = model.module
        if isinstance(model, CompositeModelBase):
            result = []
            immediate_children = []
            for name, submodel in model.submodels.items():
                subresult = ParamCountLogger._get_param_counts(submodel, trace=join_names(trace, name))
                result += subresult
                immediate_children.append(subresult[0])
            trainable_sum = sum(count for _, count, _ in immediate_children)
            frozen_sum = sum(count for _, _, count in immediate_children)
            return [(trace, trainable_sum, frozen_sum)] + result
        else:
            return [
                (
                    join_names(trace, snake_type_name(model)),
                    get_trainable_param_count(model),
                    get_frozen_param_count(model),
                )
            ]

    def _before_training(self, model, update_counter, **_):
        param_counts = self._get_param_counts(model)

        _, total_trainable, total_frozen = param_counts[0]
        max_trainable_digits = int(np.log10(total_trainable)) + 1 if total_trainable > 0 else 1
        max_frozen_digits = int(np.log10(total_frozen)) + 1 if total_frozen > 0 else 1
        # add space for thousand seperators
        max_trainable_digits += int(max_trainable_digits / 3)
        max_frozen_digits += int(max_frozen_digits / 3)
        # generate format strings
        tformat = f">{max_trainable_digits},"
        fformat = f">{max_frozen_digits},"

        self.logger.info(f"parameter counts (trainable | frozen)")
        new_summary_entries = {}
        for name, tcount, fcount in param_counts:
            name = name or "total"
            self.logger.info(f"{format(tcount, tformat)} | {format(fcount, fformat)} | {name}")
            new_summary_entries[f"param_count/{name}/trainable"] = tcount
            new_summary_entries[f"param_count/{name}/frozen"] = fcount
        self.summary_provider.update(new_summary_entries)

        # detailed number of params
        # self.logger.info("detailed parameters")
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         continue
        #     self.logger.info(f"{np.prod(param.shape):>10,} {name}")
