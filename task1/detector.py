import os
import torch
import pytorch_lightning as pl

from torch import optim, nn, utils, Tensor
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights, 
    deeplabv3_mobilenet_v3_large
)

from dataset import DocsDataset

class FreezeNetwork(pl.callbacks.Callback):
    def __init__(self, batches_to_freeze=16):
        self.batches_to_freeze = batches_to_freeze
        self.is_freezed = True
    
    def on_train_start(self, trainer, pl_module):
        for param in trainer.model.parameters():
            param.requires_grad = False
        
        for param in trainer.model.detector.classifier[-1].parameters():
            param.requires_grad = True
        for param in trainer.model.detector.aux_classifier[-1].parameters():
            param.requires_grad = True
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.is_freezed and batch_idx >= self.batches_to_freeze:
            for param in trainer.model.parameters():
                param.requires_grad = True

            self.is_freezed = False
        
class DiceLossWithLogits(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        numerator = torch.sum(inputs * targets)
        denominator = torch.sum(inputs) + torch.sum(targets)

        dice = (2. * numerator + self.alpha) / (denominator + self.alpha)

        return 1. - dice

class Detector(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.detector = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.detector.classifier[-1] = torch.nn.Conv2d(
            in_channels=self.detector.classifier[-1].in_channels,
            out_channels=1,
            kernel_size=1
        )
        self.detector.aux_classifier[-1] = torch.nn.Conv2d(
            in_channels=self.detector.aux_classifier[-1].in_channels,
            out_channels=1,
            kernel_size=1
        )

        self.critetion = DiceLossWithLogits()

    def forward(self, x):
        return self.detector(x)["out"].squeeze(1)

    def _forward_and_calc_loss(self, batch):
        x, y, _ = batch
        y = y.squeeze(1)

        x_hat = self.forward(x)
        loss = self.critetion(x_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._forward_and_calc_loss(batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_and_calc_loss(batch)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.detector.parameters(), lr=1e-3)
        return optimizer
        