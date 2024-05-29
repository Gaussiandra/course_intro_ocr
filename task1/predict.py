import torch
import numpy as np

from pathlib import Path
from torch import utils
from tqdm import tqdm

from detector import Detector
from dataset import DocsDataset

from course_intro_ocr_t1.metrics import (
    dump_results_dict, 
    measure_crop_accuracy
)

def predict(dataset_path, ckpt_path, threshold):
    test_dataset = DocsDataset(
        datapacks_path=dataset_path, 
        is_test=True, 
    )
    test_loader = utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=5,
        pin_memory=True,
        shuffle=False
    )

    detector = Detector.load_from_checkpoint(ckpt_path)
    detector.cuda().eval()

    results_dict = dict()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            x, gt_mask, unique_keys = batch

            mask = detector(x.cuda())
            # print(gt_mask.shape, mask.shape)
            mask = torch.sigmoid(mask.cpu()).numpy()
            # print(gt_mask.shape, mask.shape)

            for i, key in enumerate(unique_keys):
                mask = np.array(mask > threshold, dtype=np.uint8)

                verticies = DocsDataset.get_vertices(mask[i])
                if verticies is None:
                    continue
                    print("None!")

                verticies = verticies.astype(np.float32)
                verticies = DocsDataset.scale_vertices(
                    verticies=verticies, 
                    h=mask[i].shape[0], 
                    w=mask[i].shape[1]
                )

                results_dict[key] = verticies

                # if batch_idx < 10:
                #     import matplotlib.pyplot as plt
                #     fig, ax = plt.subplots( nrows=1, ncols=2 ) 
                #     ax[0].imshow(mask[i] > threshold)
                #     # print(mask.shape, gt_mask.shape)
                #     ax[0].scatter(verticies[:, 0] * mask[i].shape[1], verticies[:, 1] * mask[i].shape[0])
                #     ax[1].imshow(gt_mask[i][0])
                #     fig.savefig(f"viz/viz_{key}.png", dpi=400)
                #     plt.close(fig)
                # else:
                #     assert False

    dump_results_dict(results_dict, Path() / 'pred.json')
    acc = measure_crop_accuracy(
        Path() / 'pred.json',
        Path() / 'gt.json'
    )
    print("Точность кропа: {:1.4f}".format(acc))

if __name__ == "__main__":
    dataset_path = Path("/workspace/midv500_data/midv500_compressed/").resolve()
    assert dataset_path.exists(), dataset_path.absolute()

    predict(
        dataset_path=dataset_path,
        ckpt_path="lightning_logs/version_43/checkpoints/epoch=3-step=5376.ckpt",
        threshold=0.015
    )
