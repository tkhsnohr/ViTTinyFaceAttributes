import os
from glob import glob

from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    RandomHorizontalFlip,
)
import lightning as L

from .utk_dataset import FaceDataset


class FaceDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, config) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            # Get paths to training and validation images
            self.train_paths = glob(os.path.join(self.data_dir, "train", "*_*_*_*"))
            self.val_paths = glob(os.path.join(self.data_dir, "val", "*_*_*_*"))

            # Define transformations for training and validation
            self.transforms = Compose(
                [
                    RandomHorizontalFlip(),
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        elif stage == "test" or stage is None:
            # Get paths to test images
            self.test_paths = glob(os.path.join(self.data_dir, "test", "*_*_*_*"))

            # Define transformations for testing
            self.transforms = Compose(
                [
                    Resize((224, 224)),
                    ToTensor(),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDataset(self.train_paths, self.transforms),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDataset(self.val_paths, self.transforms),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            FaceDataset(self.test_paths, self.transforms),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
