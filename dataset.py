import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FGNETDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        age = self.data.iloc[idx, 1]
        
        label = 0 if age < 18 else 1

        if self.transform:
            image = self.transform(image)

        return image, label