#from __future__ import division
#import os
#import json
#import tensorflow as tf
#print(tf.__version__)
#from keras.models import load_model
#import argparse
#from preprocess import *
#import timeit
from opencxr.utils.resize_rescale import rescale_to_min_max, resize_long_edge_and_pad_to_square
import numpy as np
from opencxr.utils.mask_crop import crop_img_borders

def clip_at_percentiles(img_np, perc_low, perc_high):
    in_dtype = img_np.dtype
    low_clip = np.percentile(img_np, perc_low)
    high_clip = np.percentile(img_np, perc_high)
    return np.clip(img_np, low_clip, high_clip).astype(in_dtype)


def preprocess_img(img_x_y):
    img_x_y = rescale_to_min_max(img_x_y, np.uint8, 0, 255)

    # crop any homogeneous border regions in the image
    img_x_y, crop_values = crop_img_borders(img_x_y, in_thresh_factor=0.05)

    # clip
    img_x_y = clip_at_percentiles(img_x_y, 1, 99)

    # convert again to 8 bit unsigned
    # img_x_y = rescale_to_min_max(img_x_y, new_dtype=np.uint8, new_min=0, new_max=255)

    # resize to 256 square
    img_x_y, _, _  = resize_long_edge_and_pad_to_square(img_x_y, [1, 1], 256, pad_value=0, anti_aliasing=True, interp_order=1)

    img_x_y = rescale_to_min_max(img_x_y, new_dtype=np.uint8, new_min=0, new_max=255)


    return np.transpose(img_x_y)





