# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:11:36 2019

@author: keelin
"""
import numpy as np
import opencxr.utils
import skimage.feature
from opencxr.utils.resize_rescale import rescale_to_min_max
from scipy import ndimage
from scipy.ndimage import binary_dilation, find_objects
from skimage import measure

"""
Methods relating to mask and crop functionality
"""

def set_non_mask_constant(img_np, mask_np,
                          dilation_in_pixels=0, constant_val=0):
    """
    A method to set a portion of an image to a constant value
    e.g. set areas outside lungs to be black
    Allows for dilation of the mask if required
    :param img_np: the original image
    :param mask_np: the mask, e.g. lung mask, expected as binary 1 or 0 values
    :param dilation_in_pixels: the size of the dilation of the mask
    :param constant_val: the value to set non-mask as
    :return: The image as np with non mask areas set to the constant value
    """

    if dilation_in_pixels > 0:
        struct_element = np.ones((dilation_in_pixels,
                                  dilation_in_pixels)).astype(bool)
        mask_np = binary_dilation(mask_np,
                                  structure=struct_element
                                  ).astype(mask_np.dtype)

    np.putmask(img_np, mask_np < 1, constant_val)
    return img_np


def crop_to_mask(img_np, spacing, mask_np, margin_in_mm):
    """
    A method to crop away regions outside the bounding box of a mask
    e.g. crop to smallest rectangle containing lungs
    :param img_np: the original image
    :param spacing: The spacing of the original image
    :param mask_np: the mask to crop around, e.g. lung mask,
               expected as binary 1,0 values
    :param margin_in_mm: a margin to allow around the tightest bounding box
    :return: The cropped image
             The size changes list for reference or future use see utils __init__.py
    """
    # convert margin in mm to margin_in_pixels for each of x and y
    margin_in_pixels_x = int(np.round(margin_in_mm / spacing[0]))
    margin_in_pixels_y = int(np.round(margin_in_mm / spacing[1]))

    # get bounding box for mask
    bbox = find_objects(mask_np)
    min_x_mask = max(bbox[0][0].start - margin_in_pixels_x, 0)
    min_y_mask = max(bbox[0][1].start - margin_in_pixels_y, 0)
    max_x_mask = min(bbox[0][0].stop + margin_in_pixels_x, mask_np.shape[0])
    max_y_mask = min(bbox[0][1].stop + margin_in_pixels_y, mask_np.shape[1])

    cropped_img, size_changes = crop_with_params(img_np, [min_x_mask, max_x_mask, min_y_mask, max_y_mask])
    return cropped_img, size_changes


def crop_img_borders_by_edginess(img_np_array,
                                 width_edgy_threshold=50,
                                 dist_edgy_threshold=100):
    """
    Method to crop homogeneous border regions based on edge detection:
    1) do an edge detection to pick up edges belonging to the image content
    2) crop to the region which contains edge information
                                        (exclude small isolated edgy regions)
    :param img_np_array: the input image
    :param width_edgy_threshold: the threshold (in pixels) to define
                           an edgy region as 'small'
    :param dist_edgy_threshold: the threshold (in pixels) to define an edgy region as
                          'distant' from other edgy regions
    :return: The cropped image
             The size changes list for reference or future use see utils __init__.py
    """


    def find_starts_ends_edgy_regions_axis(edge_img_np_array, axis_to_check):
        """
        inner function to count edge pixels and identify (per row, per axis)
        where the regions with edges start and end
        """
        count_edges = []
        start_end_edgy_regions = []
        for ind in range(0, edge_img_np_array.shape[axis_to_check]):
            count_edge_pixels = np.sum(edge_img_np_array[ind, :]
                                       if axis_to_check == 0
                                       else edge_img_np_array[:, ind])
            count_edges.append(count_edge_pixels)
            if ind == 0:
                # start an edgy region if there are immediately edges present
                if count_edge_pixels > 0:
                    start_end_edgy_regions.append(ind)
                continue
            if ind == img_np_array.shape[axis_to_check] - 1:
                # if previous one was non-zero we were in an edgy region
                # so add the ending
                if count_edges[ind - 1] > 0:
                    start_end_edgy_regions.append(ind)
                # if the last row is an isolated edgy region
                # add it as a start and end also.
                elif count_edges[ind - 1] == 0 and count_edge_pixels > 0:
                    start_end_edgy_regions.append(ind)
                    start_end_edgy_regions.append(ind)
                continue

            # otherwise we are somewhere in the middle
            if count_edge_pixels > 0 and count_edges[ind - 1] == 0:
                # add start location for edgy region
                start_end_edgy_regions.append(ind)
            elif count_edge_pixels == 0 and count_edges[ind - 1] > 0:
                # add end location for edgy region
                start_end_edgy_regions.append(ind)

        return start_end_edgy_regions

    def remove_small_isolated_edgy_regions(starts_ends_edgy_regions):
        """
        inner function to remove the edgy regions which are small or isolated
        """
        # print('starts and ends are ', starts_ends_edgy_regions)
        starts_ends_retain = []
        for start_edgy_index in range(0, len(starts_ends_edgy_regions), 2):
            start_edgy = starts_ends_edgy_regions[start_edgy_index]
            end_edgy = starts_ends_edgy_regions[start_edgy_index + 1]
            print('found start and end', start_edgy, end_edgy)
            width_edgy = end_edgy - start_edgy + 1
            print('found width ', width_edgy)

            dist_next_edgy = 10000
            dist_prev_edgy = 10000
            # if a subsequent edgy region exists
            if start_edgy_index + 2 < len(starts_ends_edgy_regions):
                start_next_edgy = starts_ends_edgy_regions[start_edgy_index + 2]
                dist_next_edgy = start_next_edgy - end_edgy
            # if a previous edgy region exists
            if start_edgy_index - 2 >= 0:
                end_prev_edgy = starts_ends_edgy_regions[start_edgy_index - 1]
                dist_prev_edgy = start_edgy - end_prev_edgy

            isolated_left = (dist_prev_edgy > dist_edgy_threshold)
            isolated_right = (dist_next_edgy > dist_edgy_threshold)

            is_small_edgy_region = width_edgy < width_edgy_threshold
            is_isolated_edgy_region = isolated_left and isolated_right

            if not (is_small_edgy_region and is_isolated_edgy_region):
                starts_ends_retain.append(start_edgy)
                starts_ends_retain.append(end_edgy)
        # print('starts and ends i will retain are ', starts_ends_retain)
        return starts_ends_retain

    # Now start the main work:
    # Parameters of Canny are based on the input being in range 0-65535
    # so need to force this before we run Canny
    img_for_edge_det = rescale_to_min_max(img_np_array, new_dtype=np.uint16)

    edge_img = skimage.feature.canny(image=img_for_edge_det.astype(np.float32),
                                     sigma=5.0,
                                     low_threshold=0.0,
                                     high_threshold=500.0)
    # convert from boolean
    edge_img = edge_img.astype(np.uint8)

    # Now crop according to where the "edgy" region of the image is
    start_end_edges_x = find_starts_ends_edgy_regions_axis(edge_img, 0)
    start_end_edges_x = remove_small_isolated_edgy_regions(start_end_edges_x)

    start_end_edges_y = find_starts_ends_edgy_regions_axis(edge_img, 1)
    start_end_edges_y = remove_small_isolated_edgy_regions(start_end_edges_y)

    start_x = start_end_edges_x[0]
    end_x = start_end_edges_x[len(start_end_edges_x) - 1]
    start_y = start_end_edges_y[0]
    end_y = start_end_edges_y[len(start_end_edges_y) - 1]

    # print('Will finally crop from edginess x', start_x, end_x)
    # print('Will finally crop from edginess y', start_y, end_y)

    cropped_img, size_changes = crop_with_params(img_np_array, [start_x, end_x, start_y, end_y])
    return cropped_img, size_changes


def crop_img_borders(img_np_array, in_thresh_factor=0.05):
    """
    Method to crop homogeneous border regions based on std deviations:
    :param img_np_array: the input image
    :param in_thresh_factor: threshold factor -> 0.05 recommended
    :return: The cropped image
             The size changes list for reference or future use see utils __init__.py
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
    xmax_stored = img_np_array.shape[0] - 1
    ymin_stored = 0
    ymax_stored = img_np_array.shape[1] - 1

    # Loop until no more changes can be made
    while not completed:
        xmin = xmin_stored
        xmax = xmax_stored
        ymin = ymin_stored
        ymax = ymax_stored

        # Determine xmin
        # the first x value where the column of pixels is not "homogeneous"
        for x_pix in range(xmin_stored, xmax_stored + 1):
            line = img_np_array[x_pix, ymin_stored:ymax_stored + 1]
            std = np.std(line)
            if std < hard_threshold:
                xmin = x_pix
            else:
                break

        # Determine xmax
        # the last x value where the column of pixels is not "homogeneous"
        for x_pix in range(xmax_stored, -1, -1):
            line = img_np_array[x_pix, ymin_stored:ymax_stored + 1]
            std = np.std(line)
            if std < hard_threshold:
                xmax = x_pix
            else:
                break

        # Determine ymin
        # the first y value where the row of pixels is not "homogeneous"
        for y_pix in range(ymin_stored, ymax_stored + 1):
            line = img_np_array[xmin_stored:xmax_stored + 1, y_pix]
            std = np.std(line)
            if std < hard_threshold:
                ymin = y_pix
            else:
                break

        # Determine ymax
        # the last x value where the row of pixels is not "homogeneous"
        for y_pix in range(ymax_stored, -1, -1):
            line = img_np_array[xmin_stored:xmax_stored + 1, y_pix]
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

    cropped_img, size_changes = crop_with_params(img_np_array, [xmin_stored, xmax_stored + 1, ymin_stored, ymax_stored])
    return cropped_img, size_changes


def crop_with_params(img_np, array_minx_maxx_miny_maxy):
    """
    Crops an image accoring to the 4 params provided in the array
    :param img_np: the image to be cropped
    :param array_minx_maxx_miny_maxy: the array of crop indices min_x, max_x, min_y, max_y
    :return: The cropped image
             The size changes list for reference or future use see utils __init__.py

    """
    minx = array_minx_maxx_miny_maxy[0]
    maxx = array_minx_maxx_miny_maxy[1]
    miny = array_minx_maxx_miny_maxy[2]
    maxy = array_minx_maxx_miny_maxy[3]
    size_changes = [
        [opencxr.utils.size_change_crop_with_params, [img_np.shape[0], img_np.shape[1], minx, maxx, miny, maxy]]]
    # print('in crop_with_params, orig size', img_np.shape)
    cropped_img = img_np[minx:maxx, miny:maxy]
    # print('in crop_with_params, cropped size is ', cropped_img.shape)
    return cropped_img, size_changes


def uncrop_with_params(img_np, orig_size_x, orig_size_y, array_minx_maxx_miny_maxy, pad_value=0):
    """
    Undo a previously applied cropping operation
    :param img_np: the input image
    :param orig_size_x: the original x size (we will return to this)
    :param orig_size_y: the original y size (we will return to this)
    :param array_minx_maxx_miny_maxy: the array of crop parameters that was used
    :param pad_value: the value to pad with when we make the image larger (uncrop)
    :return: The uncropped (padded) image
             The size changes list for reference or future use see utils __init__.py
    """

    # the image being passed in was already cropped with the array info given, need to restore it
    minx = array_minx_maxx_miny_maxy[0]
    maxx = array_minx_maxx_miny_maxy[1]
    miny = array_minx_maxx_miny_maxy[2]
    maxy = array_minx_maxx_miny_maxy[3]
    pad_top = miny
    pad_bottom = orig_size_y - maxy
    pad_left = minx
    pad_right = orig_size_x - maxx

    # pad top and bottom
    img_np = np.pad(img_np, ((pad_left, pad_right), (pad_top, pad_bottom)), 'constant', constant_values=pad_value)

    # if not img_np.shape[0] == orig_size_x:
    #    print('x mismatch', img_np.shape[0], orig_size_x)
    #if not img_np.shape[1] == orig_size_y:
    #    print('y mismatch', img_np.shape[1], orig_size_y)

    size_changes = [[opencxr.utils.size_change_uncrop_with_params, [orig_size_x, orig_size_y, minx, maxx, miny, maxy]]]
    return img_np, size_changes


def get_largest_components(np_array, nr_components):
    """
    Get largest connected components (how many defined by nr_components) from a binary image, background==0
    :param np_array: a binary np array (e.g. segmentation mask)
    :param nr_components: the number of largest components to keep
    :return: a binary np array with only the largest components remaining
    """
    # get a label per connected component, background is 0
    labels = measure.label(np_array, background=0)
    # count pixels per label
    unique, counts = np.unique(labels, return_counts=True)

    # helper function
    def get_key(item):
        return item[1]

    # get ordered list of labels
    counts = sorted(zip(unique, counts), key=get_key, reverse=True)
    # get largest of labels excluding 0, and keeping only the top nr_components items
    largest_labels = [c[0] for c in counts if c[0] != 0][0:nr_components]

    if len(largest_labels) == nr_components:  # if we got the correct number of components out
        out_array = np.full(np_array.shape, False, dtype=bool)
        all_labels_np = np.asarray(labels)
        for l in largest_labels:
            out_array = out_array | (all_labels_np == l)
        out_array = out_array.astype(np.uint8)
        # print('returning correct number of components')
        return out_array
    else:  # otherwise there were not even enough components to reach nr_components so just return the original
        # print('returning fewer than expected components')
        return np_array


def tidy_segmentation_mask(seg_mask, nr_components_to_keep):
    """
    Fill holes and then keep only the largest components (e.g. 2 components for frontal lung seg, 1 component for heart seg)
    :param seg_mask: The segmentation mask to be tidied
    :param nr_components_to_keep: The number of components to keep in the final segmentation mask
    :return: The final tidied segmentation mask
    """
    # make binary
    seg_mask = rescale_to_min_max(seg_mask, np.uint8, 0, 1)

    # fill holes
    seg_mask = ndimage.binary_fill_holes(seg_mask)

    # Only keep the largest components
    seg_mask = get_largest_components(seg_mask, nr_components_to_keep).astype(np.uint8)

    seg_mask = rescale_to_min_max(seg_mask, np.uint8, 0, 255)
    return seg_mask
