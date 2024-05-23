import torch
import numpy as np
from torch.utils.data import Dataset

from course_intro_ocr_t1.data import MidvPackage


class DocsDataset(Dataset):
    def __init__(self, datapacks_path, is_test):
        self.datapacks = MidvPackage.read_midv500_dataset(datapacks_path)
        self.data_indicies = []
        self.unique_key = []
        
        for i in range(len(self.datapacks)):
            for j in range(len(self.datapacks[i])):
                if self.datapacks[i][j].is_test_split() == is_test:
                    self.data_indicies.append((i, j))
                    self.unique_key.append(self.datapacks[i][j].unique_key)

    def __len__(self):
        return len(self.data_indicies)

    def __getitem__(self, idx):
        i, j = self.data_indicies[idx]
        
        image = np.array(self.datapacks[i][j].image.convert('RGB'))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute([2, 0, 1])

        coords = self.datapacks[i][j].quadrangle.reshape(-1)

        unique_key = self.unique_key[idx]
        return image, coords, unique_key