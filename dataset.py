"""CFD (CrackForest) dataset loader with paired image/mask augmentation."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class CrackDataset(Dataset):
    def __init__(self, root, split="train", size=256, train=False):
        root = Path(root) / split
        self.images = sorted((root / "images").glob("*.jpg"))
        self.masks = [root / "masks" / (p.stem + ".png") for p in self.images]
        self.size = size
        self.train = train
        assert len(self.images) == len(self.masks) > 0, f"empty split: {root}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        msk = Image.open(self.masks[idx]).convert("L")

        img = TF.resize(img, [self.size, self.size], antialias=True)
        msk = TF.resize(msk, [self.size, self.size], interpolation=transforms.InterpolationMode.NEAREST)

        if self.train:
            if torch.rand(1).item() < 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if torch.rand(1).item() < 0.5:
                img, msk = TF.vflip(img), TF.vflip(msk)
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img, msk = TF.rotate(img, 90 * k), TF.rotate(msk, 90 * k)
            img = TF.adjust_brightness(img, float(torch.empty(1).uniform_(0.8, 1.2)))
            img = TF.adjust_contrast(img, float(torch.empty(1).uniform_(0.8, 1.2)))

        x = TF.to_tensor(img)
        x = TF.normalize(x, MEAN, STD)
        y = (TF.to_tensor(msk) > 0.5).float()
        return x, y


def split_indices(n, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * val_ratio)
    return idx[k:].tolist(), idx[:k].tolist()
