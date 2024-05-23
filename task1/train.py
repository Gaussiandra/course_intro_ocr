import torch
import pytorch_lightning as pl

from pathlib import Path
from torch import nn, utils
from tqdm import tqdm

from detector import Detector
from dataset import DocsDataset
from course_intro_ocr_t1.metrics import (
    dump_results_dict, 
    measure_crop_accuracy
)

def train(dataset_path):
    detector = Detector()

    train_dataset = DocsDataset(dataset_path, is_test=False)
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=5,
        pin_memory=True,
        shuffle=True
    )

    torch.set_float32_matmul_precision('high')
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model=detector, train_dataloaders=train_loader)

def predict(dataset_path):
    test_dataset = DocsDataset(dataset_path, is_test=True)
    test_loader = utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=5,
        pin_memory=True,
        shuffle=False
    )

    detector = Detector.load_from_checkpoint(
        "/workspace/course_intro_ocr/task1/lightning_logs/version_0/checkpoints/epoch=5-step=4032.ckpt"
    )
    detector.cuda().eval()

    results_dict = dict()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, _, unique_keys = batch
            preds = detector(x.cuda())
            
            for i, key in enumerate(unique_keys):
                results_dict[key] = preds[i].reshape(-1, 2).cpu().numpy()

    dump_results_dict(results_dict, Path() / 'pred.json')
    acc = measure_crop_accuracy(
        Path() / 'pred.json',
        Path() / 'gt.json'
    )
    print("Точность кропа: {:1.4f}".format(acc))


if __name__ == "__main__":
    dataset_path = Path("/workspace/midv500_data/midv500_compressed/").resolve()
    assert dataset_path.exists(), dataset_path.absolute()

    train(dataset_path)
    predict(dataset_path)
