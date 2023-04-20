from pathlib import Path

import yaml

from datasets.base.image_folder import ImageFolder


class ImageNet(ImageFolder):
    def __init__(self, version, split=None, **kwargs):
        self.version = version
        if version in ["imagenet_a", "imagenet_r"]:
            assert split in ["val", "test"]
        if split == "test":
            split = "val"
        assert split in ["train", "val"]
        self.split = split
        super().__init__(**kwargs)

    def get_dataset_identifier(self):
        """ returns an identifier for the dataset (used for retrieving paths from dataset_config_provider) """
        return self.version

    def get_relative_path(self):
        return Path(self.split)

    def __str__(self):
        return f"{self.version}.{self.split}"
