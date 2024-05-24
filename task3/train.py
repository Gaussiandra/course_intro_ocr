import os
import torch
import numpy as np
import pandas as pd
import csv
import pytorch_lightning as pl

from pathlib import Path
from torch import nn, utils
from tqdm import tqdm

from detector import Detector
from dataset import BarDataset


def train(ds_path, markup_path):
    detector = Detector()

    train_dataset = BarDataset(ds_path, markup_path)
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=5,
        pin_memory=True,
        shuffle=True
    )

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model=detector, train_dataloaders=train_loader)

def predict(ds_path):
    detector = Detector.load_from_checkpoint(
        "/workspace/course_intro_ocr/task3/lightning_logs/version_20/checkpoints/epoch=3-step=2052.ckpt"
    )
    detector.cuda().eval()

    results = []
    with torch.no_grad():
        for file in tqdm(os.listdir(ds_path / 'Images')):
            image, size = BarDataset._prepare_img(ds_path / "Images" / file)

            coords = detector(image.cuda().unsqueeze(0)).cpu().numpy()
            coords = coords.reshape(-1, 2)
            coords[0] *= size[0]
            coords[1] *= size[1]
            coords = np.round(coords).astype(int).reshape(-1).tolist()

            results.append([file, coords])
    
    with open('answer.csv', mode='w', encoding='utf-16') as file:
        for i in range(len(results)):
            res = ','.join([str(val) for val in results[i][1]])
            file.write(results[i][0] + ',-,' + res + ',-\n')

if __name__ == "__main__":
    train_path = Path("/workspace/bars_dataset/Train").resolve()
    markup_path = train_path / "markup.csv"
    assert train_path.exists(), train_path.absolute()
    train(train_path, markup_path)

    test_path = Path("/workspace/bars_dataset/Test").resolve()
    assert test_path.exists(), test_path.absolute()
    predict(test_path)
