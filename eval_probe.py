import os
from argparse import ArgumentParser

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_accuracy
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, Normalize, ToTensor
from tqdm import tqdm

from models.heads.linear_head import LinearHead
from models.poolings import SinglePooling
from models.vit.masked_encoder import MaskedEncoder


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/local00/bioinf/imagenet1k/val")
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--pooling", type=str, default="class_token")
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
    embed_dim = encoder_sd["cls_token"].shape[2]
    patch_size = encoder_sd["patch_embed.proj.weight"].shape[2]
    if embed_dim == 768:
        depth = 12
        attention_heads = 12
    elif embed_dim == 1024:
        depth = 24
        attention_heads = 16
    elif embed_dim == 1280:
        depth = 32
        attention_heads = 16
    else:
        raise NotImplementedError
    encoder = MaskedEncoder(
        input_shape=(3, 224, 224),
        patch_size=patch_size,
        embedding_dim=embed_dim,
        depth=depth,
        attention_heads=attention_heads,
    )
    encoder.load_state_dict(encoder_sd)
    print(f"initialize head ({head})")
    print(f"pooling {pooling}")
    head_sd = torch.load(head, map_location=torch.device("cpu"))
    if "state_dict" in head_sd:
        head_sd = head_sd["state_dict"]
    head = LinearHead(
        input_shape=encoder.output_shape,
        output_shape=(1000,),
        nonaffine_batchnorm=True,
        pooling=SinglePooling(kind=pooling),
    )
    head.load_state_dict(head_sd)
    encoder = encoder.to(device)
    head = head.to(device)
    encoder.eval()
    head.eval()

    print(f"make predictions (precision={precision})")
    preds = []
    target = []
    for x, y in tqdm(DataLoader(dataset, batch_size=256, num_workers=10, pin_memory=True)):
        with torch.no_grad():
            with torch.autocast(str(device), dtype=getattr(torch, precision)):
                preds.append(head(encoder.features(x.to(device))).cpu())
        target.append(y.clone())
    preds = torch.concat(preds)
    target = torch.concat(target)

    acc = multiclass_accuracy(
        preds=preds.to(device),
        target=target.to(device),
        num_classes=head.output_shape[0],
        average="micro",
    ).item()
    print(f"accuracy: {acc:.4f}")


if __name__ == "__main__":
    main(**parse_args())