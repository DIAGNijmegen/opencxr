# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:19:57 2020

@author: keelin
"""

import numpy as np
from opencxr.utils.mask_crop import crop_img_borders
from opencxr.utils.resize_rescale import rescale_to_min_max, resize_long_edge_and_pad_to_square


def clip_at_percentiles(img_np, perc_low, perc_high):
    """
    Clip image values based on intensity percentiles
    :param img_np: input image as np array
    :param perc_low: lower percentile (0-100)
    :param perc_high: upper percentile (0-100)
    :return: the image with clipped values
    """
    in_dtype = img_np.dtype
    low_clip = np.percentile(img_np, perc_low)
    high_clip = np.percentile(img_np, perc_high)
    return np.clip(img_np, low_clip, high_clip).astype(in_dtype)


def preprocess_img(img_x_y):
    """
    Do preprocessing before feeding to imagesorter model
    :param img_x_y: Input image as np array, x,y ordering (eg from utils file_io read_image())
    :return: the preprocessed image, prepared and transposed for the model
    """

    # convert to uint8 with min and max of 0-255
    img_x_y = rescale_to_min_max(img_x_y, np.uint8, 0, 255)

    # crop any homogeneous border regions in the image
    img_x_y, crop_values = crop_img_borders(img_x_y, in_thresh_factor=0.05)

    # clip intensities at 1st and 99th percentiles
    img_x_y = clip_at_percentiles(img_x_y, 1, 99)

    # resize to 256 square image, preserving aspect ratio
    img_x_y, _, _ = resize_long_edge_and_pad_to_square(img_x_y, [1, 1], 256, pad_value=0, anti_aliasing=True,
                                                       interp_order=1)

    # re-convert to uint8 since resize returns float
    img_x_y = rescale_to_min_max(img_x_y, new_dtype=np.uint8, new_min=0, new_max=255)

    # transpose before returning since this is what model expects
    return np.transpose(img_x_y)
