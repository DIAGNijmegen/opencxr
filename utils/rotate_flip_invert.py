# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:28:21 2019

@author: keelin
"""

from numpy import fliplr, flipud
from opencxr.utils.resize_rescale import rescale_to_min_max
from skimage import util
from skimage.transform import rotate


def invert_grayscale(np_array_in, preserve_dtype=True):
    """
    A method to invert pixel grayvalues
    If preserve_dtype is true then the result will be returned as the original
    data type.
    Resulting grey-values will be rescaled to min-max values of that dtype
    :param np_array_in: input image
    :param preserve_dtype: whether to preserve the input dtype
    :return: the image with intensities inverted
    """

    inverted_np = util.invert(np_array_in)
    if preserve_dtype:
        # cast back to original type, first rescaling to min/max for that type
        inverted_np = rescale_to_min_max(inverted_np,
                                         np_array_in.dtype,
                                         new_min=None,
                                         new_max=None, )
    return inverted_np


def rotate_img(np_array_in, rot_angle, preserve_dtype=True):
    """
    A method to rotate clockwise (rot_angle in degrees)
    If preserve_dtype is true then the result will be returned as the original
    data type.
    Resulting grey-values will be rescaled to min-max values of that dtype
    :param np_array_in: input image
    :param rot_angle: angle of rotation
    :param preserve_dtype: whether to preserve input dtype
    :return: The rotated image
    """

    rot_img = rotate(np_array_in, rot_angle)
    if preserve_dtype:
        # cast back to original type, first rescaling to min/max for that type
        rot_img = rescale_to_min_max(rot_img,
                                     np_array_in.dtype,
                                     new_min=None,
                                     new_max=None)
    return rot_img


def flip_x(np_array_in, preserve_dtype=True):
    """
    A method to flip horizontally (switch image left and right sides)
    If preserve_dtype is true then the result will be returned as the original
    data type.
    Resulting grey-values will be rescaled to min-max values of that dtype
    :param np_array_in: input image
    :param preserve_dtype: whether to preserve dtype
    :return: the horizontally flipped image
    """
    flipx = flipud(np_array_in)
    if preserve_dtype:
        # cast back to original type, first rescaling to min/max for that type
        flipx = rescale_to_min_max(flipx,
                                   np_array_in.dtype,
                                   new_min=None,
                                   new_max=None)
    return flipx


def flip_y(np_array_in, preserve_dtype=True):
    """
    A method to flip vertically (switch image top and bottom)
    If preserve_dtype is true then the result will be returned as the original
    data type.
    Resulting grey-values will be rescaled to min-max values of that dtype
    :param np_array_in: input image
    :param preserve_dtype: whether to preserve dtype
    :return: the vertically flipped image
    """
    flipy = fliplr(np_array_in)
    if preserve_dtype:
        # cast back to original type, first rescaling to min/max for that type
        flipy = rescale_to_min_max(flipy,
                                   np_array_in.dtype,
                                   new_min=None,
                                   new_max=None)
    return flipy
