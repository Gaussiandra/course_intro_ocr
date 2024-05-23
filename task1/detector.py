import os
import torch
import pytorch_lightning as pl

from torch import optim, nn, utils, Tensor
from torchvision.models import (
    mobilenet_v3_small, 
    MobileNet_V3_Small_Weights
)

from dataset import DocsDataset


class Detector(pl.LightningModule):
    def __init__(self, n_outputs=8):
        super().__init__()
        self.detector = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT,
        )
        print(self.detector)
        self.detector.classifier[-1] = nn.Linear(
            self.detector.classifier[-1].in_features,
            n_outputs
        )

    def forward(self, x):
        return self.detector(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        x_hat = self.detector(x)
        loss = nn.functional.l1_loss(x_hat, y)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.detector.parameters(), lr=1e-3)
        return optimizer
        