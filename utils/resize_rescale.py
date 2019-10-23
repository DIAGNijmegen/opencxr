# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:10:42 2018

@author: keelin
"""

from skimage.transform import resize
import numpy as np


def rescale_to_min_max(np_array_in,
                       new_dtype=None,
                       new_min=None,
                       new_max=None,
                       ):
    """
    A method to rescale array to a new min and max range.
    if new_dtype is not provided the original dtype will be assumed
    if new_min and/or new_max are not provided then the min/max possible
    for the dtype will be used.
    """
    int_types = [np.int8, np.int16, np.int32, np.int64,
                 np.uint8, np.uint16, np.uint32, np.uint64]
    float_types = [np.float32, np.float64]

    if new_dtype is None:
        new_dtype = np_array_in.dtype

    if new_dtype in int_types:
        min_poss = np.iinfo(new_dtype).min
        max_poss = np.iinfo(new_dtype).max
    elif new_dtype in float_types:
        min_poss = np.finfo(new_dtype).min
        max_poss = np.finfo(new_dtype).max
    else:
        raise Exception('Type unsupported for rescaling {}'.format(new_dtype))

    if new_min is None:
        new_min = min_poss
    if new_max is None:
        new_max = max_poss

    if new_min < min_poss or new_max > max_poss:
        raise Exception('Arrays of type {} cannot support \
                         min-max {}-{}'.format(new_dtype, new_min, new_max))

    orig_min = np.min(np_array_in)
    orig_max = np.max(np_array_in)
    std = (np_array_in - orig_min) / (orig_max - orig_min)
    scaled = std * (new_max - new_min) + new_min
    return scaled.astype(new_dtype)


def resize_isotropic(np_array_img, old_spacing, new_size_for_axis,
                     axis_specified=1, anti_aliasing=True, interp_order=1):
    """
    A method which reads in a 2D np array representing an image and resizes it
    to have isotropic spacing based on the original spacing and the new size
    along one of the dimensions.
    Parameters:
    np_array_img: The pixel values of the image in np array format
    old_spacing: The original image spacing for X and Y (2D)
    new_size_for_axis: The new image size specified for a particular axis
    axis_specified: The axis for which size is specified (0 or 1)
    anti_aliasing: Whether to use anti_aliasing in the resize operation
    interp_order: as per skimage transform:
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    e.g. resized_array, new_spacing = resize_isotropic(cxr_np_array,
                                                       pixel_spacing,
                                                       new_size_for_axis=1024,
                                                       axis_size_specified=0,
                                                       anti_aliasing=True)
    """
    orig_dtype = np_array_img.dtype
    new_spacing = (old_spacing[axis_specified] *
                   np_array_img.shape[axis_specified]) / new_size_for_axis
    other_axis = 0 if axis_specified == 1 else 1
    new_size_other_axis = round(np_array_img.shape[other_axis] *
                                old_spacing[other_axis] / new_spacing)
    resized_array = None

    if axis_specified == 0:
        resized_array = resize(np_array_img,
                               (new_size_for_axis, new_size_other_axis),
                               preserve_range=True,
                               mode='reflect',
                               anti_aliasing=anti_aliasing,
                               order=interp_order)
    else:
        resized_array = resize(np_array_img,
                               (new_size_other_axis, new_size_for_axis),
                               preserve_range=True,
                               mode='reflect',
                               anti_aliasing=anti_aliasing,
                               order=interp_order)
    resized_array = resized_array.astype(orig_dtype)
    return np.squeeze(resized_array), [new_spacing, new_spacing]


def resize_to_x_y(np_array_img, old_spacing, new_size_0, new_size_1,
                  anti_aliasing=True, interp_order=1):
    """
    A method which reads in a 2D np array representing an image and
    resizes to new x y dimensions.
    Parameters:
    np_array_img: The pixel values of the image in np array format
    old_spacing: The original image spacing for X and Y (2D)
    new_size_0: The new image size on axis 0
    new_size_1: The new image size on axis 1
    anti_aliasing: Whether to use anti_aliasing in the resize operation
    e.g. resized_array, new_spacing = resize_to_x_y(cxr_np_array,
                                                       pixel_spacing,
                                                       new_size_0=1024,
                                                       new_size_1=2048,
                                                       anti_aliasing=True)
    """
    new_img = resize(image=np_array_img, output_shape=(new_size_0, new_size_1),
                     preserve_range=True, mode='reflect')
    new_spacing_x = old_spacing[0] * np_array_img.shape[0] / new_size_0
    new_spacing_y = old_spacing[1] * np_array_img.shape[1] / new_size_1

    return new_img, [new_spacing_x, new_spacing_y]
