from course_intro_ocr_t1.metrics import read_results_dict
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

pred = read_results_dict("pred.json")
gt = read_results_dict("gt.json")

for i, k in enumerate(gt.keys()):
    if i > 10:
        break

    fig, ax = plt.subplots( nrows=1, ncols=2 ) 
    # ax[0].imshow(mask[i] > threshold)
    # print(mask.shape, gt_mask.shape)
    gt_verticies = np.array(gt[k])
    pred_verticies = np.array(pred[k])

    ax[0].scatter(pred_verticies[:, 0], pred_verticies[:, 1])
    ax[1].scatter(gt_verticies[:, 0], gt_verticies[:, 1])
    fig.savefig(f"json_viz/viz_{k}.png", dpi=400)