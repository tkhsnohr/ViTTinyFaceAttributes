import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class FaceDataset(Dataset):
    def __init__(self, img_list: list[str], transforms: Compose):
        super(Dataset, self).__init__()

        self.img_list = img_list  # List of image paths
        self.transforms = transforms  # Image transformations to be applied

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]  # Path of the image
        img = Image.open(img_path)  # Open the image using PIL
        img = self.transforms(img)  # Apply the specified transformations to the image

        # Extract age, gender, and ethnicity from the image filename
        filename = os.path.basename(img_path)  # Get the filename from the image path
        # Extract the age from the filename
        age = torch.tensor([int(filename.split("_")[0])])
        # Extract the gender from the filename
        gender = torch.tensor(int(filename.split("_")[1]))
        # Extract the ethnicity from the filename
        ethnicity = torch.tensor(int(filename.split("_")[2]))

        return img, age, gender, ethnicity
