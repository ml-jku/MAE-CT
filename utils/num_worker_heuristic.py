import os

from distributed.config import is_slurm_run


# noinspection PyUnusedLocal
def get_num_workers(dataset, batch_size, n_datasets, is_train_dataset, run_type):
    cpu_count = get_fair_cpu_count()
    if cpu_count == 0:
        return 0

    # use less than cpu_count (too many workers often run into errors)
    # - "OSError: [Errno 24] Too many open files"
    # - "RuntimeError: Too many open files. Communication with the workers is no longer possible. ..."
    # max_workers = int(cpu_count / 2)
    max_workers = cpu_count - 1

    # eval runs shouldn't use prefetching -> use full workers
    if run_type == "eval":
        return max_workers

    # distribute workers among train/test datasets
    if n_datasets > 1:
        # use a lot of workers for train dataset
        n_train_workers = int(max_workers * 0.75)
        if is_train_dataset:
            return n_train_workers
        else:
            # check if eval run
            if n_datasets == 1:
                return n_train_workers
            # distribute workers among test datasets
            return max(4, int((max_workers - n_train_workers) / (n_datasets - 1)))
    else:
        # only train dataset -> use all workers
        return max_workers


def get_num_fetch_workers(dataset):
    return 0


def get_fair_cpu_count():
    total_cpu_count = get_total_cpu_count()
    if total_cpu_count == 0:
        return 0

    device_count = _get_device_count()
    # divide cpus among devices
    if is_slurm_run():
        # slurm already divides cpus among tasks -> assert that
        tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        cpus_per_task = int(os.environ["SLURM_CPUS_PER_TASK"])
        # currently only 1 GPU per task is supported
        assert device_count == tasks_per_node
        assert total_cpu_count == cpus_per_task
        # use 75% of slurm workers for dataloading
        # 16worker MAE-B 512bs/A100 -> 0.05 data time
        # 24worker MAE-B 512bs/A100 -> 0.00 data time
        return total_cpu_count - 1
    return int(total_cpu_count / device_count)


def _get_device_count():
    # get number of devices per node (srun nvidia-smi shows all devices not only the ones assigned for the srun task)
    # (if no GPU is available this returns "")
    # normal example output:
    # GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    # GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    # MIG example output:
    # GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    #   MIG 3g.20gb     Device  0: (UUID: MIG-...)
    #   MIG 3g.20gb     Device  1: (UUID: MIG-...)
    # GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-...)
    #   MIG 3g.20gb     Device  0: (UUID: MIG-...)
    #   MIG 3g.20gb     Device  1: (UUID: MIG-...)
    nvidia_smi_lines = os.popen("nvidia-smi -L").read().strip().split("\n")

    # create dict from GPU to MIG devices:
    # {
    #   GPU0: 1 # normal GPU
    #   GPU1: 2 # split into 2 MIG devices
    # }
    devices_per_gpu = {}
    devices_counter = 0
    for i, line in enumerate(nvidia_smi_lines):
        if "MIG" in line:
            devices_counter += 1
        if "GPU" in line and i == 0 and len(nvidia_smi_lines) > 1 and "MIG" in nvidia_smi_lines[i + 1]:
            continue
        if "GPU" in line or i == len(nvidia_smi_lines) - 1:
            if devices_counter == 0:
                devices_counter = 1  # normal GPU -> single device
            devices_per_gpu[len(devices_per_gpu)] = devices_counter
            devices_counter = 0
    # count devices
    devices_on_node = sum(devices_per_gpu.values())

    if devices_on_node == 0:
        devices_on_node = 1
    return devices_on_node


def get_total_cpu_count():
    if os.name == "nt":
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        if cpu_count <= 16:
            # don't bother on dev machines
            return 0
    else:
        cpu_count = len(os.sched_getaffinity(0))

    return cpu_count
