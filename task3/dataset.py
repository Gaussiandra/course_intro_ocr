import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, io
from pathlib import Path


class BarDataset(Dataset):
    def __init__(self, ds_path, markup_path):
        self.COORDS = slice(2, 10)

        self.df = pd.read_csv(markup_path, encoding='utf-16', header=None)
        self.path = Path(ds_path)

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _prepare_img(img_path):
        pil_image = Image.open(img_path)
        image = np.asarray(pil_image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute([2, 0, 1])
        image = transforms.Resize((256, 256))(image)

        return image, pil_image.size
    
    def __getitem__(self, idx):
        img_path = self.path / "Images" / self.df.iloc[idx][0]
        image, size = BarDataset._prepare_img(img_path)

        target = np.array(self.df.iloc[idx][self.COORDS], dtype=np.float32).reshape(-1, 2)
        target[:, 0] /= size[0]
        target[:, 1] /= size[1]
        target = target.reshape(-1)
        target = torch.tensor(target)
                
        return image, target
