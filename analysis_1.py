import MAL.losses.segment as l
import MAL.phantom as ph

# List of various loss functions to evaluate
loss_functions = [
    l.dice(),
    l.iou(),
    l.jaccard(),
    l.weighted_binary_cross_entropy(),
    l.mean_squred_error(),
    l.mean_absolute_error(),
    l.mean_error(),
    l.tversky(),
    l.focal_tversky(),
    l.hausdorff(),
]

# List of ground truth phantom shapes
ground_truth_shapes = [
    ph.Ones(n_classes=1),
    ph.Ones(n_classes=1),
    ph.Zeros(n_classes=1),
    ph.Rectangle(n_classes=1),
    ph.Circle(n_classes=1),
    ph.Oval(n_classes=1),
]

# List of predicted output phantom shapes
pred_output_shapes = [
    ph.Ones(n_classes=1),
    ph.Zeros(n_classes=1),
    ph.Ones(n_classes=1),
    ph.Rectangle(n_classes=1, random_remove=[1]),
    ph.Circle(n_classes=1, random_remove=[1]),
    ph.Oval(n_classes=1, random_remove=[1]),
]

# Loop over ground truth and predicted shapes
for ground_truth_shape, pred_output_shape in zip(ground_truth_shapes, pred_output_shapes):
    # Loop over different loss functions
    for loss_function in loss_functions:
        # Generate ground truth and predicted output
        ground_truth = ground_truth_shape()
        pred_output = pred_output_shape()

        # Calculate loss value using the selected loss function
        loss_value = loss_function(
            ground_truth,
            pred_output,
        )

        # Get the used functions in ground truth and predicted shapes
        used_func_gt = ground_truth_shape.used_functions()
        used_func_pr = pred_output_shape.used_functions()

        # Print the used functions for ground truth and predicted shapes if available
        # if used_func_gt:
        #     print(f'Used functions in Ground Truth: {used_func_gt}')
        # if used_func_pr:
        #     print(f'Used functions in Predicted Output: {used_func_pr}')

        # Print the loss value along with the names of loss function, ground truth, and predicted output shapes
        print(f'Loss: {loss_function.name} \t GT: {ground_truth_shape.name()} \t PO: {pred_output_shape.name()} \t:\t {loss_value}')

    # Print a newline for better readability between different ground truth and predicted output shapes
    print('\n')
# NOTE: excel save