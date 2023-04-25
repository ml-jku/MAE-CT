import logging
import sys

import kappaconfig
import kappadata
import kappaprofiler
import packaging.version
import pytorch_concurrent_dataloader
import torch
import torchmetrics

expected_kappaconfig = "1.0.29"
expected_kappadata = "1.0.99"
expected_kappaprofiler = "1.0.9"
# torchmetrics 0.11.0 is backward compatibility breaking
# accuracy method requires task parameter or using multiclass_accuracy instead of just calling accuracy(pred, targets)
# accuracy(task="multiclass", ...)
expected_torchmetrics_version = "0.11.0"
expected_python_major = 3
expected_python_minor = 7


def check_versions(verbose):
    log_fn = logging.info if verbose else lambda _: None

    log_fn("------------------")
    log_fn("VERSION CHECK")

    # python version >= 3.7 for order preserving dict (https://docs.python.org/3/whatsnew/3.7.html)
    py_version = sys.version_info
    msg = f"upgrade python ({py_version.major}.{py_version.minor} < {expected_python_major}.{expected_python_minor})"
    assert py_version.major >= expected_python_major and py_version.minor >= expected_python_minor, msg
    log_fn(f"python version: {py_version.major}.{py_version.minor}.{py_version.micro}")

    #
    log_fn(f"torch version: {torch.__version__}")
    log_fn(f"torchmetrics version: {torchmetrics.__version__}")

    def _check_pip_dependency(actual_version, expected_version, pip_dependency_name):
        assert packaging.version.parse(actual_version) >= packaging.version.parse(expected_version), (
            f"upgrade {pip_dependency_name} with 'pip install {pip_dependency_name} --upgrade' "
            f"({actual_version} < {expected_version})"
        )
        log_fn(f"{pip_dependency_name} version: {actual_version}")

    _check_pip_dependency(kappaconfig.__version__, expected_kappaconfig, "kappaconfig")
    _check_pip_dependency(kappadata.__version__, expected_kappadata, "kappadata")
    _check_pip_dependency(kappaprofiler.__version__, expected_kappaprofiler, "kappaprofiler")
    _check_pip_dependency(torchmetrics.__version__, expected_torchmetrics_version, "torchmetrics")
