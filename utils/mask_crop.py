# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:11:36 2019

@author: keelin
"""
import numpy as np
from scipy.ndimage import binary_dilation, find_objects


def set_non_mask_constant(img_np, mask_np,
                          dilation_in_pixels=0, constant_val=0):
    """
    A method to set a portion of an image to a constant value
    e.g. set areas outside lungs to be black
    Allows for dilation of the mask if required
    Parameters:
    img_np - the original image
    mask_np - the mask, e.g. lung mask, expected as binary 1 or 0 values
    dilation_in_pixels - the size of the dilation of the mask
    constant_val - the value to set non-mask as
    Returns:
    The image as np with non mask areas set to the constant value
    """
    if dilation_in_pixels > 0:
        struct_element = np.ones((dilation_in_pixels,
                                  dilation_in_pixels)).astype(bool)
        mask_np = binary_dilation(mask_np,
                                  structure=struct_element
                                  ).astype(mask_np.dtype)

    np.putmask(img_np, mask_np < 1, constant_val)
    return img_np


def crop_to_mask(img_np, mask_np, margin_in_pixels):
    """
    A method to crop away regions outside the bounding box of a mask
    e.g. crop to smallest rectangle containing lungs
    Parameters:
    img_np - the original image
    mask_np - the mask to crop around, e.g. lung mask,
               expected as binary 1,0 values
    margin_in_pixels - a margin to allow around the tightest bounding box
    """
    # get bounding box for mask
    bbox = find_objects(mask_np)
    min_x_mask = max(bbox[0][0].start - margin_in_pixels, 0)
    min_y_mask = max(bbox[0][1].start - margin_in_pixels, 0)
    max_x_mask = min(bbox[0][0].stop + margin_in_pixels, mask_np.shape[0] - 1)
    max_y_mask = min(bbox[0][1].stop + margin_in_pixels, mask_np.shape[1] - 1)

    crop_img = img_np[min_x_mask:max_x_mask, min_y_mask:max_y_mask]

    return crop_img


def crop_img_borders(img_np_array, in_thresh_factor=0.05):
    """
    A method which reads in a cxr image (numpy) and crops homogeneous
    regions around the edges
    Parameters:
    img_np_array: 2D np array representing the image
    in_thresh_factor: the threshold factor to start with - recommended 0.05
    Returns:
    The cropped image and the bounding box coords [xmin, xmax, ymin, ymax]
    e.g. new_img, crop_vals = crop_img_borders(img_np, 0.05)
    """
    xmin = 0
    xmax = img_np_array.shape[0]
    ymin = 0
    ymax = img_np_array.shape[1]

    # Use the image std dev and in_thresh_factor to make a hard threshold
    img_std_dev = np.std(img_np_array)
    hard_threshold = in_thresh_factor * img_std_dev

    # Some setup
    completed = False
    xmin_stored = 0
    xmax_stored = img_np_array.shape[0]-1
    ymin_stored = 0
    ymax_stored = img_np_array.shape[1]-1

    # Loop until no more changes can be made
    while not completed:
        xmin = xmin_stored
        xmax = xmax_stored
        ymin = ymin_stored
        ymax = ymax_stored

        # Determine xmin
        # the first x value where the column of pixels is not "homogeneous"
        for x_pix in range(xmin_stored, xmax_stored+1):
            line = img_np_array[x_pix, ymin_stored:ymax_stored+1]
            std = np.std(line)
            if std < hard_threshold:
                xmin = x_pix
            else:
                break

        # Determine xmax
        # the last x value where the column of pixels is not "homogeneous"
        for x_pix in range(xmax_stored, -1, -1):
            line = img_np_array[x_pix, ymin_stored:ymax_stored+1]
            std = np.std(line)
            if std < hard_threshold:
                xmax = x_pix
            else:
                break

        # Determine ymin
        # the first y value where the row of pixels is not "homogeneous"
        for y_pix in range(ymin_stored, ymax_stored+1):
            line = img_np_array[xmin_stored:xmax_stored+1, y_pix]
            std = np.std(line)
            if std < hard_threshold:
                ymin = y_pix
            else:
                break

        # Determine ymax
        # the last x value where the row of pixels is not "homogeneous"
        for y_pix in range(ymax_stored, -1, -1):
            line = img_np_array[xmin_stored:xmax_stored+1, y_pix]
            std = np.std(line)
            if std < hard_threshold:
                ymax = y_pix
            else:
                break

        # In case of any unusual image causing issues - just keep stored values
        if xmax < xmin:
            xmin = xmin_stored
        if ymax < ymin:
            ymin = ymin_stored

        # Assume that we are done, but if we made any change on this
        # last loop then we will set completed=False and try one more time
        completed = True
        if xmin > xmin_stored:
            xmin_stored = xmin
            completed = False
        if xmax < xmax_stored:
            xmax_stored = xmax
            completed = False
        if ymin > ymin_stored:
            ymin_stored = ymin
            completed = False
        if ymax < ymax_stored:
            ymax_stored = ymax
            completed = False

    # crop the image to the relevant sub-image
    out_img_np_array = img_np_array[xmin_stored:xmax_stored+1,
                                    ymin_stored:ymax_stored]
    # return the image and the crop values
    return out_img_np_array, [xmin, xmax, ymin, ymax]
