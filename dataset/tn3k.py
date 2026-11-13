"""TN3K thyroid-nodule segmentation dataset adapter for AF-CLIP."""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TN3KDataset(Dataset):
    """Load the official TN3K test split from ``root/tn3k``.

    The test split contains test-image/*.jpg and test-mask/*.jpg with matching
    stems. TN3K provides lesion masks but no normal test images, so every
    sample has image-level label 1 and is intended for pixel-level evaluation.
    """

    def __init__(self, root, train=False, category=None, transform=None,
                 gt_target_transform=None):
        self.root = Path(root) / "tn3k"
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.train = train
        self.categories = ["thyroid"]
        self.category = "thyroid"
        self.dataset_name = "tn3k"
        self._load_test()

    def _load_test(self):
        image_dir = self.root / "test-image"
        mask_dir = self.root / "test-mask"
        image_paths = sorted(p for p in image_dir.iterdir() if p.is_file())
        mask_map = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
        missing = [p.name for p in image_paths if p.stem not in mask_map]
        if missing:
            raise FileNotFoundError(
                f"TN3K masks missing for {len(missing)} images, e.g. {missing[:3]}"
            )
        self.img_paths = image_paths
        self.gt_paths = [mask_map[p.stem] for p in image_paths]
        self.labels = [1] * len(image_paths)

    def update(self, category=None):
        self.category = "thyroid"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.gt_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        if self.gt_target_transform is not None:
            mask = self.gt_target_transform(mask)
        return image, self.labels[idx], mask, self.category, str(img_path)
