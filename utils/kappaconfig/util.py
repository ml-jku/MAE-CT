import logging
import shutil
from pathlib import Path

import kappaconfig as kc
import yaml

from .mindata_postprocessor import MinDataPostProcessor
from .mindata_preprocessor import MinDataPreProcessor
from .minduration_postprocessor import MinDurationPostProcessor
from .minmodel_preprocessor import MinModelPreProcessor
from .precision_preprocessor import PrecisionPreProcessor
from .remove_large_collections_postprocessor import RemoveLargeCollectionsProcessor


def _get_hp_file_uri(hp_file):
    file_uri = Path(hp_file).expanduser().with_suffix(".yaml")
    assert file_uri.exists(), f"hp_file '{file_uri}' doesn't exist"
    return file_uri


def save_unresolved_hp(hp_file, out_file_uri):
    file_uri = _get_hp_file_uri(hp_file)
    shutil.copy(file_uri, out_file_uri)
    logging.info(f"copied unresolved hp to {out_file_uri}")


def save_resolved_hp(stage_hp, out_file_uri):
    stage_hp = remove_large_collections(stage_hp)
    with open(out_file_uri, "w") as f:
        yaml.safe_dump(stage_hp, f)
    logging.info(f"dumped resolved hp to {out_file_uri}")


def get_run_hp(hp_file):
    file_uri = _get_hp_file_uri(hp_file)
    run_hp = kc.from_file_uri(file_uri)
    return run_hp


def get_stage_hp_list(
        variant_hp,
        template_path=None,
        testrun=False,
        minmodelrun=False,
        mindatarun=False,
        mindurationrun=False,
):
    resolver = kc.DefaultResolver(template_path=template_path)
    resolver.pre_processors.append(PrecisionPreProcessor())
    if minmodelrun or testrun:
        resolver.pre_processors.append(MinModelPreProcessor())
    if mindatarun or testrun:
        resolver.pre_processors.append(MinDataPreProcessor())
        resolver.post_processors.append(MinDataPostProcessor())
    if mindurationrun or testrun:
        resolver.post_processors.append(MinDurationPostProcessor())

    if "stages" not in variant_hp:
        return [None], [resolver.resolve(variant_hp)]

    stages = variant_hp.pop("stages")
    stage_names = []
    stage_hp_list = []
    for stage_name, stage_hp in stages.items():
        merged = kc.merge(variant_hp, stage_hp, allow_path_accessors=True)
        resolved = resolver.resolve(merged)
        stage_names.append(stage_name)
        stage_hp_list.append(resolved)
    return stage_names, stage_hp_list


def remove_large_collections(stage_hp):
    stage_hp = kc.from_primitive(stage_hp)
    resolver = kc.Resolver(post_processors=[RemoveLargeCollectionsProcessor()])
    resolved = resolver.resolve(stage_hp)
    return resolved


def log_stage_hp(stage_hp):
    stage_hp = remove_large_collections(stage_hp)
    yaml_str = yaml.safe_dump(stage_hp, sort_keys=False)
    # safe_dump appends a trailing newline
    logging.info(f"------------------\n{yaml_str[:-1]}")


def get_stage_ids_from_cli():
    """
    for starting multi-stage runs the already finished stage_ids have to be provided from the cli
    e.g. stage0=pretrain stage1=probe -> only start probe -> python main_train.py ... stage_ids.pretrain=k234o1l23
    """
    cli_args = kc.from_cli()
    stage_ids = kc.mask_in(cli_args, "stage_ids")
    if "stage_ids" in stage_ids:
        return kc.to_primitive(stage_ids)["stage_ids"]
    else:
        return {}


def get_max_batch_sizes_from_cli():
    """
    allow to define max_batch_sizes for all stages
    e.g. stage0=pretrain stage1=probe -> python main_train.py ... max_batch_sizes.pretrain=16 max_batch_sizes.probe=64
    """
    cli_args = kc.from_cli()
    max_batch_sizes = kc.mask_in(cli_args, "max_batch_sizes")
    if "max_batch_sizes" in max_batch_sizes:
        return kc.to_primitive(max_batch_sizes)["max_batch_sizes"]
    else:
        return {}
