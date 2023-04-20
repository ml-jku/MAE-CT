from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS, find_classes


class TVImageFolder(DatasetFolder):
    """
    same as torchvision.datasets.folder.ImageFolder but allows passing a custom class_to_idx
    used for ImageNet-A/ImageNet-R which only have 200 classes and their class indices
    don't correspond to the original ImageNet1K class indices
    """

    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
            class_to_idx=None,
    ):
        self.custom_class_to_idx = class_to_idx
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def find_classes(self, directory: str):
        classes, class_to_idx = find_classes(directory)
        if self.custom_class_to_idx is not None:
            class_to_idx = self.custom_class_to_idx
        return classes, class_to_idx
