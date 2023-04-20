# python eval_knn_mlp.py --encoder ".../models/ibot_vitb_rand_teacher.pth" --device 0
import torch.nn as nn
from argparse import ArgumentParser
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, Normalize, ToTensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
import torch
from torchmetrics.functional.classification import multiclass_accuracy
from tqdm import tqdm
from models.vit.vit_mae import VitMae
from models.vit.masked_encoder import MaskedEncoder
from models.heads.linear_head import LinearHead
from models.poolings import SinglePooling
import os
from pathlib import Path
from metrics.functional.knn import knn_metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=".../imagenet1k")
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return vars(parser.parse_args())


def main(root, encoder, device, disable_flash_attention, precision):
    root = Path(root).expanduser()
    encoder = Path(encoder).expanduser()
    print(f"initialize dataset ({root})")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if disable_flash_attention:
        os.environ["DISABLE_FLASH_ATTENTION"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        Resize(size=256, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size=224),
        ToTensor(),
        Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    train_dataset = ImageFolder(root=root / "train", transform=transform)
    print(f"train dataset has {len(train_dataset)} samples and {len(train_dataset.classes)} classes")
    test_dataset = ImageFolder(root=root / "val", transform=transform)
    print(f"train dataset has {len(test_dataset)} samples and {len(test_dataset.classes)} classes")

    print(f"initialize encoder ({encoder})")
    encoder_sd = torch.load(encoder, map_location=torch.device("cpu"))
    # MAE checkpoints
    if "model" in encoder_sd:
        encoder_sd = encoder_sd["model"]
    # MLPlayground/IBOT checkpoints
    if "state_dict" in encoder_sd:
        encoder_sd = encoder_sd["state_dict"]
    dim, channels, patch_height, patch_width = encoder_sd["patch_embed.proj.weight"].shape
    depth = max(int(key.split(".")[1]) for key in encoder_sd.keys() if key.startswith("blocks.")) + 1
    if depth > 12:
        attn_heads = 16
    elif dim == 768:
        attn_heads = 12
    else:
        raise NotImplementedError
    encoder = MaskedEncoder(
        input_shape=(channels, 224, 224),
        patch_size=(patch_height, patch_width),
        embedding_dim=dim,
        depth=depth,
        attention_heads=attn_heads,
    )
    encoder.load_state_dict(encoder_sd)
    encoder = encoder.to(device)
    encoder.eval()

    print(f"extract train features (precision={precision} disable_flash_attention={disable_flash_attention})")
    train_x = []
    train_y = []
    for x, y in tqdm(DataLoader(train_dataset, batch_size=256, num_workers=10, pin_memory=True)):
        with torch.no_grad():
            with torch.autocast(str(device), dtype=getattr(torch, precision)):
                train_x.append(encoder.features(x.to(device)).cpu())
        train_y.append(y.clone())
    train_x = torch.concat(train_x)
    train_y = torch.concat(train_y)
    
    print(f"extract test features (precision={precision} disable_flash_attention={disable_flash_attention})")
    test_x = []
    test_y = []
    for x, y in tqdm(DataLoader(test_dataset, batch_size=256, num_workers=10, pin_memory=True)):
        with torch.no_grad():
            with torch.autocast(str(device), dtype=getattr(torch, precision)):
                test_x.append(encoder.features(x.to(device)).cpu())
        test_y.append(y.clone())
    test_x = torch.concat(test_x)
    test_y = torch.concat(test_y)

    print(f"calculate knn (knn=20)")
    accuracies, _, _ = knn_metrics(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, knns=[20])
    acc = accuracies[0]
    print(f"accuracy: {acc:.4f}")
    print(f"accuracy: {acc:.8f}")



if __name__ == "__main__":
    main(**parse_args())