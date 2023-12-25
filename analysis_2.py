import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import MAL.losses.segment as l
import MAL.phantom as ph
import MAL.utils as utils

# List of various loss functions to evaluate
loss_functions = [
    l.dice(),
    l.iou(),
    l.jaccard(),
    # l.weighted_binary_cross_entropy(),
    l.mean_squred_error(),
    l.mean_absolute_error(),
    l.mean_error(),
    l.tversky(),
    l.focal_tversky(),
    # l.hausdorff(smooth=0.001),
]

# List of ground truth phantom shapes
ground_truth_shapes = [
    ph.Rectangle(n_classes=1),
]

# List of predicted output phantom shapes
pred_output_shapes = [
    ph.Rectangle(n_classes=1),
]

# Loop over the pairs of ground truth and predicted shapes
for idx, (pr, gt) in enumerate(zip(pred_output_shapes, ground_truth_shapes)):
    # Dictionary to store the output of each loss function
    output = {key.name: [] for key in loss_functions}

    # List to store generated images during the iteration
    images = []

    # Progress bar to track the iteration progress
    pbar = tqdm(range(0, pr.count() + 1), desc=f'{len(np.nonzero(pr())[0])}')

    # Iterate through different configurations by removing a random subset of elements
    for i in pbar:
        # Generate ground truth and predicted output images
        gt_image = gt()
        pr_image = pr()

        # Update the predicted output by removing a random subset of elements
        pr.update(random_remove=[i])

        # Calculate and store the loss value for each loss function
        for loss_function in loss_functions:
            loss_value = loss_function(gt_image, pr_image)
            output[loss_function.name].append(loss_value.numpy())

        # Update the progress bar description with the count of non-zero elements in the predicted output
        pbar.desc = f'{len(np.nonzero(pr())[0])}'

    # Plot and save the normalized loss curves for each loss function
    fig = plt.figure()
    for jdx, loss_function in enumerate(loss_functions):
        ax = fig.add_subplot(3, 3, jdx+1)
        series_loss_values = output[loss_function.name]
        normalized_series = utils.min_max_normalization(series_loss_values)
        ax.plot(list(range(0, pr.count() + 1)), normalized_series)
        ax.set_title(loss_function.name)
        print(f'{loss_function.name} : \t\t {np.min(output[loss_function.name])}, {np.max(output[loss_function.name])}')

    # Add legend and save the figure
    fig.set_tight_layout('rect')
    fig.savefig(f'analysis_2_{idx}.pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for loss_function in loss_functions:
        series_loss_values = output[loss_function.name]
        normalized_series = utils.min_max_normalization(series_loss_values)
        ax.plot(list(range(0, pr.count() + 1)), normalized_series)
        print(f'{loss_function.name} : \t\t {np.min(output[loss_function.name])}, {np.max(output[loss_function.name])}')

    # Add legend and save the figure
    ax.legend([l.name for l in loss_functions])
    fig.set_tight_layout('rect')
    
    fig.savefig(f'analysis_2_1_{idx}.pdf')
    plt.close()