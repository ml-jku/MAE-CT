import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, Normalize, ToTensor
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/local00/bioinf/imagenet1k/val")
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--pooling", type=str, default="token", choices=["token", "avg"])
    return vars(parser.parse_args())


def main(root, encoder, head, device, precision, pooling):
    print(f"initialize dataset ({root})")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder(
        root=root,
        transform=Compose([
            Resize(size=256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size=224),
            ToTensor(),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]),
    )
    assert len(dataset.classes) == 1000

    print(f"initialize encoder ({encoder})")
    encoder_sd = torch.load(encoder, map_location=torch.device("cpu"))
    if "state_dict" in encoder_sd:
        encoder_sd = encoder_sd["state_dict"]
    dim = encoder_sd["pos_embed"].shape[2]
    if dim == 768:
        model = vit_base_patch16_224(use_fc_norm=False, global_pool=pooling)
    elif dim == 1024:
        model = vit_large_patch16_224(use_fc_norm=False, global_pool=pooling)
    elif dim == 1280:
        patch_size = encoder_sd["patch_embed.proj.weight"].shape[2]
        if patch_size == 16:
            model = vit_huge_patch14_224(use_fc_norm=False, global_pool=pooling, patch_size=16)
        elif patch_size == 14:
            model = vit_huge_patch14_224(use_fc_norm=False, global_pool=pooling)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # timm ViT has a zero vector for the pos_embed of cls token
    encoder_sd["pos_embed"] = torch.concat([torch.zeros(1, 1, model.embed_dim), encoder_sd["pos_embed"]], dim=1)
    print(f"initialize head ({head})")
    head_sd = torch.load(head, map_location=torch.device("cpu"))
    if "state_dict" in head_sd:
        head_sd = head_sd["state_dict"]
    # patch head (probing heads use a non-affine batchnorm before the linear layer)
    if "layer.1.running_mean" in head_sd:
        model.head = nn.Sequential(
            nn.BatchNorm1d(num_features=model.embed_dim, affine=False),
            nn.Linear(model.embed_dim, model.num_classes),
        )
        encoder_sd["head.0.running_mean"] = head_sd["layer.1.running_mean"]
        encoder_sd["head.0.running_var"] = head_sd["layer.1.running_var"]
        encoder_sd["head.0.num_batches_tracked"] = head_sd["layer.1.num_batches_tracked"]
        encoder_sd["head.1.weight"] = head_sd["layer.2.weight"]
        encoder_sd["head.1.bias"] = head_sd["layer.2.bias"]
    else:
        encoder_sd["head.weight"] = head_sd["layer.2.weight"]
        encoder_sd["head.bias"] = head_sd["layer.2.bias"]
    model.load_state_dict(encoder_sd)
    model = model.to(device)
    model.eval()

    print(f"make predictions (precision={precision})")
    preds = []
    target = []
    for x, y in tqdm(DataLoader(dataset, batch_size=256, num_workers=10, pin_memory=True)):
        with torch.no_grad():
            with torch.autocast(str(device), dtype=getattr(torch, precision)):
                preds.append(model(x.to(device)).cpu())
        target.append(y.clone())
    preds = torch.concat(preds)
    target = torch.concat(target)

    acc = multiclass_accuracy(
        preds=preds.to(device),
        target=target.to(device),
        num_classes=model.num_classes,
        average="micro",
    ).item()
    print(f"accuracy: {acc:.4f}")


if __name__ == "__main__":
    main(**parse_args())
