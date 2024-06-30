import torch
from torch import nn
import numpy as np
import math
import cv2


def ndarray_to_tensor(x, is_cuda=False, requires_grad=False, dtype=torch.float32):
    t = torch.tensor(x, dtype=dtype, requires_grad=requires_grad)
    if is_cuda:
        t = t.cuda()
    return t


def caluate_game(est, gt, L):
    # assert len(gt.shape
    if len(gt.shape) == 2:
        gt = gt[np.newaxis, np.newaxis, :, :]
        est = est[np.newaxis, np.newaxis, :, :]
    gt = ndarray_to_tensor(gt, is_cuda=False)
    est = ndarray_to_tensor(est, is_cuda=False)
    width, height = gt.shape[3], gt.shape[2]
    times = L
    padding_height = int(math.ceil(height / times) * times - height)
    padding_width = int(math.ceil(width / times) * times - width)
    if padding_height != 0 or padding_width != 0:
        m = nn.ZeroPad2d((0, padding_width, 0, padding_height))
        gt = m(gt)
        est = m(est)
        width, height = gt.shape[3], gt.shape[2]

    m = nn.AdaptiveAvgPool2d(int(times))

    gt = m(gt) * (height / times) * (width / times)
    est = m(est) * (height / times) * (width / times)
    mae = torch.sum(torch.abs(gt - est)) / (times ** 2)
    mse = torch.sum((gt - est) * (gt - est)) / (times ** 2)

    return mae.item(), mse.item()


def calculate_fb_error(est, gt, mask, log_factor):
    # forground_mask = np.zeros_like(gt)
    # forground_mask[gt > 0] = 1
    k_s = 7
    forground_mask = mask.astype(np.float)
    # print(forground_mask.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_s, k_s))
    forground_mask = cv2.dilate(forground_mask, kernel)
    # forground_mask = cv2.erode(forground_mask, kernel)
    background_mask = 1 - forground_mask
    forground_density_est = forground_mask * est
    # forground_density_gt = forground_mask * gt

    f_mae = abs(np.sum(forground_density_est) / log_factor - np.sum(gt) / log_factor)

    background_density_est = background_mask * est
    # background_density_gt = background_mask * gt
    background_density_gt = 0

    b_mae = abs(np.sum(background_density_est) / log_factor - np.sum(background_density_gt))

    return f_mae, b_mae


