# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:10:42 2018

@author: keelin
"""
import numpy as np
import opencxr
from skimage.transform import resize


def rescale_to_min_max(np_array_in,
                       new_dtype=None,
                       new_min=None,
                       new_max=None,
                       ):
    """
    A method to rescale array values (image intensities) to a new min and max range.
    if new_dtype is not provided the original dtype will be assumed
    if new_min and/or new_max are not provided then the min/max possible
    for the dtype will be used.
    :param np_array_in: array to rescale
    :param new_dtype: new dtype if required
    :param new_min: new min if required
    :param new_max: new max if required
    :return: The rescaled np array
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
    :param np_array_img: Input array
    :param old_spacing: Original image spacing
    :param new_size_for_axis: The new image size specified for a particular axis
    :param axis_specified: The axis for which size is specified (0 or 1)
    :param anti_aliasing: Whether to use anti_aliasing in the resize operation
    :param interp_order: as per skimage transform 0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    :return: The resized array
             The new spacing
              The list of size changes carried out for reference or future use (see utils __init__.py)
    """

    orig_dtype = np_array_img.dtype
    new_spacing = (old_spacing[axis_specified] *
                   np_array_img.shape[axis_specified]) / new_size_for_axis
    other_axis = 0 if axis_specified == 1 else 1
    new_size_other_axis = round(np_array_img.shape[other_axis] *
                                old_spacing[other_axis] / new_spacing)
    resized_array = None

    if axis_specified == 0:
        resized_array, returned_spacing, size_changes = resize_to_x_y(np_array_img,
                                                                      old_spacing,
                                                                      new_size_for_axis,
                                                                      new_size_other_axis,
                                                                      anti_aliasing=anti_aliasing,
                                                                      order=interp_order)
    else:
        resized_array, returned_spacing, size_changes = resize_to_x_y(np_array_img,
                                                                      old_spacing,
                                                                      new_size_other_axis,
                                                                      new_size_for_axis,
                                                                      anti_aliasing=anti_aliasing,
                                                                      order=interp_order)

    resized_array = resized_array.astype(orig_dtype)
    return np.squeeze(resized_array), returned_spacing, size_changes


def resize_to_x_y(np_array_img, old_spacing, new_size_0, new_size_1,
                  anti_aliasing=True, interp_order=1):
    """
     A method which reads in a 2D np array representing an image and
    resizes to new x y dimensions.
    :param np_array_img: Input image
    :param old_spacing: original image spacing
    :param new_size_0: new size for axis 0
    :param new_size_1: new size for axis 1
    :param anti_aliasing:  Whether to use anti_aliasing in the resize operation
    :param interp_order: as per skimage transform 0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    :return: The new resized image
             The new image spacing
             The list of size changes carried out for reference or future use (see utils __init__.py)

    """

    new_img = resize(image=np_array_img, output_shape=(new_size_0, new_size_1),
                     preserve_range=True, mode='reflect', anti_aliasing=anti_aliasing,
                     order=interp_order)
    new_spacing_x = old_spacing[0] * np_array_img.shape[0] / new_size_0
    new_spacing_y = old_spacing[1] * np_array_img.shape[1] / new_size_1

    size_changes = [[opencxr.utils.size_change_resize_to_x_y,
                     [np_array_img.shape[0], np_array_img.shape[1], new_size_0, new_size_1]]]

    return new_img, [new_spacing_x, new_spacing_y], size_changes


def resize_preserve_aspect_ratio(np_array_img, old_spacing, new_size_for_axis, axis_specified, anti_aliasing=True,
                                 interp_order=1):
    """
     A method to resize an image preserving the aspect ratio.  One new edge size is specified and the axis
    that is to be resized to that size.  The other dimension will be resized accordingly.
    :param np_array_img: The input image
    :param old_spacing: The original image spacing
    :param new_size_for_axis: The length being specified for one axis
    :param axis_specified: The axis to be resized to the given length
    :param anti_aliasing: Whether to use anti-aliasing
    :param interp_order:  as per skimage transform 0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    :return: The resized image
             The new image spacing
             The list of size changes carried out for reference or future use (see utils __init__.py)
    """

    if axis_specified == 0:
        mult_factor = new_size_for_axis / np_array_img.shape[0]
        new_shape_0 = new_size_for_axis
        new_shape_1 = int(np.round(np_array_img.shape[1] * mult_factor))
    else:
        mult_factor = new_size_for_axis / np_array_img.shape[1]
        new_shape_0 = int(np.round(np_array_img.shape[0] * mult_factor))
        new_shape_1 = new_size_for_axis

    img_out, new_spacing, size_changes = resize_to_x_y(np_array_img, old_spacing, new_shape_0, new_shape_1,
                                                       anti_aliasing, interp_order)

    return img_out, new_spacing, size_changes


def resize_long_edge_and_pad_to_square(np_array_img, old_spacing, square_edge_size, pad_value=0, anti_aliasing=True,
                                       interp_order=1):
    """
     A method to resize an image to a square size (square_edge_size * square_edge_size) without changing aspect ratio
    The image is first resized, preserving aspect ratio, such that the longer edge is of length square_edge_size
    The shorter edge is then padded (split across left/right or top/bottom) to bring it to the same length.
    :param np_array_img: the input image
    :param old_spacing: the input image spacing
    :param square_edge_size: the output image required edge length
    :param pad_value: the intensity value for padding pixels
    :param anti_aliasing: whether to use anti-aliasing
    :param interp_order: as per skimage transform 0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    :return: The resized image
             The new image spacing
             The list of size changes carried out for reference or future use (see utils __init__.py)
    """

    shape_x, shape_y = np_array_img.shape

    if shape_x >= shape_y:
        # do resize
        img_resized, new_spacing, resize_size_changes = resize_preserve_aspect_ratio(np_array_img, old_spacing,
                                                                                     square_edge_size, 0, anti_aliasing,
                                                                                     interp_order)

        # pad smaller dimension (in this case top and bottom)
        diff = square_edge_size - img_resized.shape[1]
        img_padded, pad_size_changes = pad_axis_with_total(img_resized, axis=1, total_pad=diff)

    else:
        # do resize
        img_resized, new_spacing, resize_size_changes = resize_preserve_aspect_ratio(np_array_img, old_spacing,
                                                                                     square_edge_size, 1, anti_aliasing,
                                                                                     interp_order)

        # pad smaller dimension (in this case left and right)
        diff = square_edge_size - img_resized.shape[0]
        img_padded, pad_size_changes = pad_axis_with_total(img_resized, axis=0, total_pad=diff)

    # append the pad_size changes on to the resize_size_changes before returning
    resize_size_changes.extend(pad_size_changes)

    return img_padded, new_spacing, resize_size_changes


def pad_axis_with_total(img_np, axis, total_pad, pad_value=0):
    """
    Pad an image with rows (if axis==1) or columns (if axis==0)
    The total number of rows/columns will be split between the top/bottom or the left/right of the image
    :param img_np: The input image
    :param axis: The axis to pad  (0 pads left/right, 1 pads top/bottom)
    :param total_pad: The total number of rows/columns to be add.  Will be split in half.
    :param pad_value: The value to give to padding rows/cols
    :return: The padded image
             The list of size changes carried out for reference or future use (see utils __init__.py)
    """

    # pad top and bottom
    if axis == 1:
        top_pad = int(np.round(float(total_pad) / 2))
        bottom_pad = total_pad - top_pad
        img_np = np.pad(img_np, ((0, 0), (top_pad, bottom_pad)), 'constant', constant_values=pad_value)

    elif axis == 0:
        left_pad = int(np.round(float(total_pad) / 2))
        right_pad = total_pad - left_pad
        img_np = np.pad(img_np, ((left_pad, right_pad), (0, 0)), 'constant', constant_values=pad_value)

    return img_np, [[opencxr.utils.size_change_pad_axis_with_total, [axis, total_pad, pad_value]]]


def un_pad_axis_with_total(img_np, axis, total_pad):
    """
    Reverse a previous padding of an image removing rows (if axis==1) or columns (if axis==0)
    The total number of rows/columns removed will be split between the top/bottom or the left/right of the image
    :param img_np: the input image
    :param axis: the axis to (un)pad (0 pads left/right, 1 pads top/bottom)
    :param total_pad: The total number of rows/columns to be un-padded.  Will be split in half.
    :return: The unpadded image
             The list of size changes carried out for reference or future use (see utils __init__.py)
    """
    # unpad top and bottom
    if axis == 1:
        top_pad = int(np.round(float(total_pad) / 2))
        bottom_pad = total_pad - top_pad
        full_height = img_np.shape[1]
        img_np = img_np[:, top_pad:full_height - bottom_pad]

    # unpad left and right
    elif axis == 0:
        left_pad = int(np.round(float(total_pad) / 2))
        right_pad = total_pad - left_pad
        full_width = img_np.shape[0]
        img_np = img_np[left_pad:full_width - right_pad, :]

    return img_np, [[opencxr.utils.size_change_unpad_axis_with_total, [axis, total_pad]]]
