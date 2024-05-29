import cv2
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from course_intro_ocr_t1.data import MidvPackage


class DocsDataset(Dataset):
    def __init__(self, datapacks_path, is_test):
        self.is_test = is_test

        self.datapacks = MidvPackage.read_midv500_dataset(datapacks_path)
        self.data_indicies = []
        self.unique_key = []
        
        for i in range(len(self.datapacks)):
            for j in range(len(self.datapacks[i])):
                if self.datapacks[i][j].is_test_split() == self.is_test:
                    self.data_indicies.append((i, j))
                    self.unique_key.append(self.datapacks[i][j].unique_key)
        
        self.norm = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )


    def __len__(self):
        return len(self.data_indicies)

    def __getitem__(self, idx):
        i, j = self.data_indicies[idx]
        
        image = self.datapacks[i][j].image.convert('RGB')
        w, h = image.size
        image = torch.tensor(np.array(image), dtype=torch.float32)
        image = image.permute([2, 0, 1])
        image = self.norm(image)

        background = np.zeros((h, w))
        poly = np.array(self.datapacks[i][j].gt_data['quad'])
        mask = cv2.fillConvexPoly(background, poly, 1)
        mask = torch.tensor(mask).unsqueeze(0)

        if not self.is_test:
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

        unique_key = self.unique_key[idx]
        return image, mask, unique_key
    
    @staticmethod
    def get_vertices(mask):
        assert mask.dtype == np.uint8

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contour = max(contours, key=cv2.contourArea)
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        convex_indices = np.array([
            np.argmin(x + y), np.argmin(-x + y), 
            np.argmax(x + y), np.argmax(-x + y)
        ])

        verticies = np.array([x[convex_indices], y[convex_indices]]).T

        return verticies

    @staticmethod
    def scale_vertices(verticies, w, h):
        verticies = verticies.copy()
        verticies[:, 0] /= w
        verticies[:, 1] /= h

        return verticies
        