import torch
import pytorch_lightning as pl

from pathlib import Path
from torch import nn, utils

from detector import Detector, FreezeNetwork
from dataset import DocsDataset

def train(dataset_path):
    detector = Detector()

    train_dataset = DocsDataset(
        datapacks_path=dataset_path, 
        is_test=False, 
    )

    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=5,
        pin_memory=True,
        shuffle=True
    )

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[
            FreezeNetwork(50),
        ]
    )
    trainer.fit(
        model=detector,
        train_dataloaders=train_loader,
    )

if __name__ == "__main__":
    dataset_path = Path("/workspace/midv500_data/midv500_compressed/").resolve()
    assert dataset_path.exists(), dataset_path.absolute()

    train(dataset_path)
