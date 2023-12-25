import random
import numpy as np
import tensorflow as tf

import MAL.losses.segment as l
import MAL.phantom as ph
import MAL.utils as utils

import matplotlib.pyplot as plt
from tqdm import tqdm

loss_functions = [
    l.hausdorff(),
    l.dice(),
    l.iou(),
    l.jaccard(),
    # l.weighted_binary_cross_entropy(),
    l.mean_squred_error(),
    l.mean_absolute_error(),
    l.mean_error(),
    l.tversky(),
    l.focal_tversky(),
]

ground_truth_shapes = [
    ph.Rectangle(n_classes=1),
]

pred_output_shapes = [
    ph.Rectangle(n_classes=1),
]

for idx, (pr, gt) in enumerate(zip(pred_output_shapes, ground_truth_shapes)):
    output = {key.name : [] for key in loss_functions}

    images = []

    _left = int(1)
    _right = int(128)

    pbar = tqdm(range(_left, _right), desc=f'{len(np.nonzero(pr())[0])}')
    for i in pbar:
        pr.update(
            shape_width=i,
            shape_height=i,
        )

        gt_image = gt()
        pr_image = pr()
        images.append(pr_image)

        for loss_function in loss_functions:
            loss_value = loss_function(
                gt_image,
                pr_image,
            )
            output[loss_function.name].append(loss_value)

        # if i == _right/2+1:
        #     plt.imshow(gt_image[0])
        #     plt.show()
        #     a = tf.convert_to_tensor(pr_image, dtype=tf.float64)
        #     a = l.create_contour_from_tensor(a)
        #     plt.imshow(a[0])
        #     plt.show()
        #     print(output['Hausdorff'])

        pbar.desc = f'{len(np.nonzero(pr())[0])}'
    
    # utils.create_gif(images)

    fig = plt.figure()
    for jdx, loss_function in enumerate(loss_functions):
        ax = fig.add_subplot(3, 3, jdx+1)
        series_loss_values = output[loss_function.name]
        normalized_series = series_loss_values
        ax.plot(list(range(_left, _right)), normalized_series)
        ax.set_title(loss_function.name)
        print(f'{loss_function.name} : \t\t {np.min(output[loss_function.name])}, {np.max(output[loss_function.name])}')

    # Add legend and save the figure
    fig.set_tight_layout('rect')
    
    fig.savefig(f'analysis_6_{idx}.pdf')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for loss_function in loss_functions:
        series_loss_values = output[loss_function.name]
        normalized_series = utils.min_max_normalization(series_loss_values)
        ax.plot(list(range(_left, _right)), normalized_series)
        print(f'{loss_function.name} : \t\t {np.min(output[loss_function.name])}, {np.max(output[loss_function.name])}')

    # Add legend and save the figure
    ax.legend([l.name for l in loss_functions])
    fig.set_tight_layout('rect')
    
    fig.savefig(f'analysis_6_{idx+1}.pdf')
    plt.close()

    # fig = plt.figure()
    # for jdx, loss_function in enumerate(loss_functions):
    #     ax = fig.add_subplot(3, 3, jdx+1)
    #     series_loss_values = output[loss_function.name]
    #     normalized_series = utils.z_score_normalization(series_loss_values, mean=0.0, std=1.5)
    #     ax.plot(list(range(_left, _right)), normalized_series)
    #     ax.set_title(loss_function.name)
    #     print(f'{loss_function.name} : \t\t {np.min(output[loss_function.name])}, {np.max(output[loss_function.name])}')

    # # Add legend and save the figure
    # fig.set_tight_layout('rect')
    
    # fig.savefig(f'analysis_6_{idx+2}.pdf')
    # plt.close()

