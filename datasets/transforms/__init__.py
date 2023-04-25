from kappadata.transforms import KDComposeTransform

from utils.factory import instantiate


def transform_collate_fn(objs):
    return KDComposeTransform(objs)


def transform_from_kwargs(kind, **kwargs):
    # torchvision.transforms
    tv_transform = instantiate(
        module_names=[
            f"datasets.transforms.{kind}",
            "torchvision.transforms",
            f"kappadata.transforms.{kind}",
            f"kappadata.transforms.norm.{kind}",
            "kappadata.common.transforms",
        ],
        type_names=[kind],
        error_on_not_found=False,
        **kwargs,
    )
    if tv_transform is not None:
        return tv_transform

    # custom transforms

    raise RuntimeError(f"unknown transform function {kind}")
