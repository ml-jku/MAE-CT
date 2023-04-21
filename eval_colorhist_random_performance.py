# for ImageNet-1K validation set:
# r: 0.011546174064278603
# g: 0.01179442647844553
# b: 0.012775475159287453
# overall: 0.012038691900670528
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from kappadata import color_histogram
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, ToTensor
from tqdm import tqdm

from losses.functional.color_histogram_losses import color_histogram_regression_loss


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=".../imagenet1k/val")
    parser.add_argument("--device", type=int)
    parser.add_argument("--perfect_prediction", action="store_true")
    return vars(parser.parse_args())


def main(root, device, perfect_prediction):
    root = Path(root).expanduser()
    print(f"initialize dataset ({root})")
    if device is None:
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        device = torch.device("cuda")
    dataset = ImageFolder(
        root=root,
        transform=Compose([
            Resize(size=256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size=224),
            ToTensor(),
        ]),
    )
    assert len(dataset.classes) == 1000

    losses = []
    for x, _ in tqdm(DataLoader(dataset, batch_size=128, num_workers=10, pin_memory=True)):
        x = x.to(device, non_blocking=True) * 255
        if perfect_prediction:
            pred = color_histogram(x, bins=64, density=True)
        else:
            pred = torch.zeros(len(x), 192, device=device)
        loss = color_histogram_regression_loss(
            preds=pred,
            images=x,
            bins=64,
            loss_fn=l1_loss,
            reduction="none",
            temperature=None if perfect_prediction else 1.,
        )
        losses.append(loss.cpu())
    losses = torch.concat(losses)
    print(losses.shape)
    print(f"r: {losses[:, 0].mean()}")
    print(f"g: {losses[:, 1].mean()}")
    print(f"b: {losses[:, 2].mean()}")
    print(f"overall: {losses.mean()}")


if __name__ == "__main__":
    main(**parse_args())
