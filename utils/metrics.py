"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models
"""
import numpy as np
from skimage.filters import threshold_otsu


def thresholding(preds, gt):
    """
    Thresholding function that converts gt and preds into binary images
    Activated prediction pixels are found via Otsu's thresholding function
    """
    N, T = gt.shape[0], gt.shape[1]
    res = np.zeros_like(preds)

    # For each sample and timestep, get Otsu's threshold and binarize gt and pred
    for n in range(N):
        for t in range(T):
            img = preds[n, t]
            otsu_th = np.max([0.32, threshold_otsu(img)])
            res[n, t] = (img > otsu_th).astype(np.float32)
            gt[n, t] = (gt[n, t] > 0.55).astype(np.float32)
    return res, gt


def dst(gts, preds, **kwargs):
    """
    Computes a Euclidean distance metric between the center of the ball in ground truth and prediction
    Activated pixels in the predicted are computed via Otsu's thresholding function
    :param gts: ground truth sequences
    :param preds: model predicted sequences
    """
    # Ensure on CPU and numpy
    if not isinstance(gts, np.ndarray):
        gts = gts.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()

    # Get shapes
    num_samples, timesteps, height, width = gts.shape

    # For each sample and timestep, get Otsu's threshold and binarize gts and pred
    gts = (gts > 0.55).astype(np.float32)
    preds = (preds > threshold_otsu(preds)).astype(np.float32)

    # Loop over each sample and timestep to get the distance metric
    results = np.zeros([num_samples, timesteps])
    for n in range(num_samples):
        pred, gt = preds[n], gts[n]

        pred_pos = np.stack(np.where(pred == 1), axis=1)
        gt_pos = np.stack(np.where(gt == 1), axis=1)

        # Split into each timestep
        pred_groups = np.split(pred_pos[:, 1:], np.unique(pred_pos[:, 0], return_index=True)[1][1:])
        gt_groups = np.split(gt_pos[:, 1:], np.unique(gt_pos[:, 0], return_index=True)[1][1:])

        # Pad if prediction doesn't make it all the way to 20 timesteps
        if len(pred_groups) < len(gt_groups):
            for _ in range(len(gt_groups) - len(pred_groups)):
                pred_groups.append(np.array([[0, 0]]))

        gt_centers = np.array([(gpos[:, 0].mean(), gpos[:, 1].mean()) for gpos in gt_groups])
        pred_centers = np.array([(ppos[:, 0].mean(), ppos[:, 1].mean()) for ppos in pred_groups])

        # Get distance metric
        dist = np.sum((pred_centers - gt_centers) ** 2, axis=1)
        dist = np.sqrt(dist)

        # Add to result
        results[n] = dist

    return results, np.mean(results), np.std(results)


def recon_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['args'].gen_len] - target[:, :kwargs['args'].gen_len]) ** 2
    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2, 3))
    return sequence_pixel_mse, np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def extrapolation_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for a number of steps past the length used in training """
    full_pixel_mses = (output[:, kwargs['args'].gen_len:] - target[:, kwargs['args'].gen_len:]) ** 2
    if full_pixel_mses.shape[1] == 0:
        return 0.0, 0.0

    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2, 3))
    return sequence_pixel_mse, np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)
