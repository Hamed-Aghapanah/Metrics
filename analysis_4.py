import random
import numpy as np

import MAL.losses.segment as l
import MAL.phantom as ph
import MAL.utils as utils

import matplotlib.pyplot as plt
from tqdm import tqdm

loss_functions = [
    l.focal_tversky(),
    l.focal_tversky(alpha=0.8, name='FocalTversky alpha=0.8'),
    l.focal_tversky(alpha=0.7, name='FocalTversky alpha=0.7'),
    l.focal_tversky(alpha=0.6, name='FocalTversky alpha=0.6'),
    l.focal_tversky(alpha=0.5, name='FocalTversky alpha=0.5'),
    l.focal_tversky(alpha=0.4, name='FocalTversky alpha=0.4'),
    l.focal_tversky(alpha=0.4, name='FocalTversky alpha=0.3'),
    l.focal_tversky(alpha=0.4, name='FocalTversky alpha=0.2'),
    l.focal_tversky(alpha=0.4, name='FocalTversky alpha=0.1'),
]

ground_truth_shapes = [
    ph.Rectangle(n_classes=1),
]

pred_output_shapes = [
    ph.Rectangle(n_classes=1, value = 'gauss'),
]

for idx, (pr, gt) in enumerate(zip(pred_output_shapes, ground_truth_shapes)):
    output = {key.name : [] for key in loss_functions}

    pbar = tqdm(range(1, pr.count()), desc=f'{len(np.nonzero(pr())[0])}')
    for i in pbar:
        pr.update(random_remove=[i])
        for loss_function in loss_functions:
            gt_image = gt()
            pr_image = pr()
            loss_value = loss_function(
                gt_image,
                pr_image,
            )
            output[loss_function.name].append(loss_value)
        pbar.desc = f'{len(np.nonzero(pr())[0])}'
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    for loss_function in loss_functions:
        ax.plot(list(range(1, pr.count())), output[loss_function.name])
        print(f'{loss_function.name} : \t {np.min(output[loss_function.name])}, {np.max(output[loss_function.name])}')
    ax.legend([l.name for l in loss_functions])
    ax = fig.add_subplot(1, 2, 2)
    for loss_function in loss_functions:
        ax.plot(list(range(1, pr.count())), utils.min_max_normalization(output[loss_function.name]))
    ax.legend([l.name for l in loss_functions])
    fig.set_tight_layout('rect')
    fig.savefig(f'analysis_4_{idx}.pdf')
    plt.close()
