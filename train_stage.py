import os
import logging
from pathlib import Path

import kappaprofiler as kp
import yaml

from configs.cli_args import CliArgs
from configs.static_config import StaticConfig
from configs.util import cliarg_or_staticvalue
from configs.wandb_config import WandbConfig
from datasets import dataset_from_kwargs
from distributed.config import is_rank0, is_distributed, get_rank, log_distributed_config
from loggers.base.logger_base import LoggerBase
from loggers.stage_summarizers import stage_summarizer_from_kwargs
from loggers.summary_summarizers import summary_summarizer_from_kwargs
from loggers.writers.log_writer import LogWriter
from loggers.writers.noop_writer import NoopWriter
from models import model_from_kwargs
from providers.dataset_config_provider import DatasetConfigProvider
from providers.stage_path_provider import StagePathProvider
from trainers import trainer_from_kwargs
from utils.commands import command_from_kwargs
from utils.data_container import DataContainer
from utils.kappaconfig.util import save_unresolved_hp, save_resolved_hp, log_stage_hp
from utils.logging_util import add_global_handlers
from utils.memory_leak_util import get_tensors_in_memory
from utils.seed import set_seed
from utils.system_info import log_system_info, get_cli_command
from utils.version_check import check_versions
from utils.wandb_utils import init_wandb, finish_wandb


def train_stage(
        stage_hp: dict,
        static_config: StaticConfig,
        cliargs: CliArgs,
        device: str,
        stage_name: str,
        stage_id: str,
        previous_stage_ids: dict,
        max_batch_size: int,
):
    # initialize logging
    stage_path_provider = StagePathProvider(
        output_path=static_config.output_path,
        model_path=static_config.model_path,
        stage_name=stage_name,
        stage_id=stage_id,
        previous_stage_ids=previous_stage_ids,
    )
    message_counter = add_global_handlers(log_file_uri=stage_path_provider.logfile_uri)

    # overwrite accelerator if it is defined from yaml
    if "accelerator" in stage_hp and stage_hp["accelerator"] == "cpu":
        device = "cpu"

    # initialize wandb
    wandb_config_uri = stage_hp.pop("wandb", None)
    if wandb_config_uri == "disabled":
        wandb_mode = "disabled"
    else:
        wandb_mode = cliarg_or_staticvalue(cliargs.wandb_mode, static_config.default_wandb_mode)
    if wandb_mode == "disabled":
        wandb_config_dict = {}
        if cliargs.wandb_config is not None or wandb_config_uri is not None:
            logging.warning(f"wandb_config is defined via CLI but mode is disabled -> wandb_config is not used")
    else:
        # retrieve wandb config from yaml
        if wandb_config_uri is not None:
            wandb_config_uri = Path("wandb_configs") / wandb_config_uri
            if cliargs.wandb_config is not None:
                logging.warning(f"wandb_config is defined via CLI and via yaml -> wandb_config from yaml is used")
        # retrieve wandb config from --wandb_config cli arg
        elif cliargs.wandb_config is not None:
            wandb_config_uri = Path("wandb_configs") / cliargs.wandb_config
        # use default wandb_config file
        else:
            wandb_config_uri = Path("wandb_config.yaml")
        with open(wandb_config_uri.with_suffix(".yaml")) as f:
            wandb_config_dict = yaml.safe_load(f)
    wandb_config = WandbConfig(mode=wandb_mode, **wandb_config_dict)
    config_provider, summary_provider = init_wandb(
        device=device,
        run_name=cliargs.name or stage_hp.pop("name", None),
        stage_hp=stage_hp,
        resume_id=cliargs.wandb_resume_id,
        wandb_config=wandb_config,
        stage_path_provider=stage_path_provider,
        account_name=static_config.account_name,
        tags=stage_hp.pop("tags", None),
        notes=stage_hp.pop("notes", None),
        group=stage_hp.pop("group", None),
        group_tags=stage_hp.pop("group_tags", None),
    )

    # flash attention
    if stage_hp.pop("disable_flash_attention", False) or cliargs.disable_flash_attention:
        os.environ["DISABLE_FLASH_ATTENTION"] = "true"

    # log setup
    logging.info("------------------")
    logging.info(get_cli_command())
    check_versions(verbose=True)
    log_system_info()
    cliargs.log()
    log_distributed_config()
    log_stage_hp(stage_hp)
    if is_rank0():
        save_unresolved_hp(cliargs.hp, stage_path_provider.stage_output_path / "hp_unresolved.yaml")
        save_resolved_hp(stage_hp, stage_path_provider.stage_output_path / "hp_resolved.yaml")
    logging.info("------------------")
    logging.info(f"training stage '{stage_path_provider.stage_name}'")

    seed = stage_hp.get("seed", None)
    if seed is None:
        logging.info(f"no seed specified -> using seed=5")
        seed = 5
    if is_distributed():
        # using a different seed for every rank to ensure that stochastic processes are different across ranks
        # for large batch_sizes this shouldn't matter too much
        # this is relevant for:
        # - augmentations (augmentation parameters of sample0 of rank0 == augparams of sample0 of rank1 == ...)
        # - the masks of a MAE are the same for every rank
        # NOTE: DDP syncs the parameters in its __init__ method -> same initial parameters independent of seed
        seed += get_rank()
        logging.info(f"using different seeds per process (seed+rank) ")
    set_seed(seed)

    # init datasets
    logging.info("------------------")
    logging.info("initializing datasets")
    datasets = {}
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static_config.get_global_dataset_paths(),
        local_dataset_path=static_config.get_local_dataset_path(),
        data_source_modes=static_config.get_data_source_modes(),
        data_caching_modes=static_config.get_data_caching_modes(),
    )
    for dataset_key, dataset_kwargs in stage_hp["datasets"].items():
        logging.info(f"initialzing {dataset_key}")
        datasets[dataset_key] = dataset_from_kwargs(dataset_config_provider=dataset_config_provider, **dataset_kwargs)
    data_container = DataContainer(**datasets, num_workers=cliargs.num_workers, config_provider=config_provider)

    # init logwriter
    if is_rank0():
        LoggerBase.log_writer_singleton = LogWriter(stage_path_provider=stage_path_provider)
    else:
        LoggerBase.log_writer_singleton = NoopWriter()

    # init trainer
    logging.info("------------------")
    logging.info("initializing trainer")
    trainer_kwargs = {}
    if max_batch_size is not None:
        trainer_kwargs["max_batch_size"] = max_batch_size
    trainer = trainer_from_kwargs(
        data_container=data_container,
        device=device,
        sync_batchnorm=cliarg_or_staticvalue(cliargs.sync_batchnorm, static_config.default_sync_batchnorm),
        config_provider=config_provider,
        summary_provider=summary_provider,
        stage_path_provider=stage_path_provider,
        **stage_hp["trainer"],
        **trainer_kwargs,
    )

    # init model
    logging.info("------------------")
    logging.info("creating model")
    model = model_from_kwargs(
        **stage_hp["model"],
        input_shape=trainer.input_shape,
        output_shape=trainer.output_shape,
        update_counter=trainer.update_counter,
        stage_path_provider=stage_path_provider,
    )
    # moved to trainer as initialization on cuda is different than on cpu
    # model = model.to(stage_config.run_config.device)

    # train model
    trainer.train(model)

    # finish loggers
    LoggerBase.finish()

    # summarize logvalues
    logging.info("------------------")
    logging.info(f"summarize logvalues")
    summary_provider.summarize_logvalues()

    # summarize stage
    if "stage_summarizers" in stage_hp and is_rank0():
        logging.info("------------------")
        logging.info("summarize stage")
        for kwargs in stage_hp["stage_summarizers"]:
            summarizer = stage_summarizer_from_kwargs(
                summary_provider=summary_provider,
                stage_path_provider=stage_path_provider,
                **kwargs,
            )
            summarizer.summarize()
    # summarize summary
    if "summary_summarizers" in stage_hp and is_rank0():
        summary_provider.flush()
        logging.info("------------------")
        for kwargs in stage_hp["summary_summarizers"]:
            summary_summarizer = summary_summarizer_from_kwargs(
                summary_provider=summary_provider,
                **kwargs,
            )
            summary_summarizer.summarize()
    summary_provider.flush()

    # add profiling times to summary_provider
    def try_log_profiler_time(summary_key, profiler_query):
        try:
            summary_provider[summary_key] = kp.profiler.get_node(profiler_query).total_time
        except AssertionError:
            pass

    try_log_profiler_time("profiler/train", "train")
    try_log_profiler_time("profiler/train/iterator", "train.iterator")
    try_log_profiler_time("profiler/train/data_loading", "train.data_loading")
    try_log_profiler_time("profiler/train/update", "train.update")
    try_log_profiler_time("profiler/train/to_device", "train.update.forward.to_device")
    try_log_profiler_time("profiler/train/forward", "train.update.forward")
    try_log_profiler_time("profiler/train/backward", "train.update.backward")
    summary_provider.flush()
    # log profiler times
    logging.info(f"full profiling times:\n{kp.profiler.to_string()}")
    kp.reset()

    # execute commands
    if "on_finish" in stage_hp and is_rank0():
        logging.info("------------------")
        logging.info("ON_FINISH COMMANDS")
        for command in stage_hp["on_finish"]:
            command = command_from_kwargs(**command, stage_id=stage_id)
            # noinspection PyBroadException
            try:
                command.execute()
            except:
                logging.exception(f"failed to execute {command}")

    # cleanup
    logging.info("------------------")
    logging.info(f"CLEANUP")
    data_container.dispose()
    message_counter.log()
    finish_wandb(wandb_config)

    # log how many tensors remain to be aware of potential memory leaks
    all_tensors, cuda_tensors = get_tensors_in_memory()
    logging.info("------------------")
    logging.info(f"{len(all_tensors)} tensors remaining in memory (cpu+gpu)")
    logging.info(f"{len(all_tensors) - len(cuda_tensors)} tensors remaining in memory (cpu)")
    logging.info(f"{len(cuda_tensors)} tensors remaining in memory (gpu)")
