from opencxr.utils.mask_crop import crop_with_params, uncrop_with_params
from opencxr.utils.resize_rescale import resize_to_x_y, un_pad_axis_with_total, pad_axis_with_total

size_change_resize_to_x_y = 'sc_resize_to_x_y'
size_change_pad_axis_with_total = 'sc_pad_axis_with_total'
size_change_unpad_axis_with_total = 'sc_unpad_axis_with_total'
size_change_crop_with_params = 'sc_crop_with_params'
size_change_uncrop_with_params = 'sc_uncrop_with_params'


def apply_size_changes_to_img(img_np, spacing, size_changes, anti_aliasing=True, interp_order=1):
    """
    Applies specified size changes to an image.  Algorithms which change size (e.g cxr standardization) can return size_changes information in list format
    This method is a utility to allow users to apply those same size changes to another image.
    Args:
        img_np: the input image in x,y axis ordering
        size_changes: the size changes returned from an algorithm, to be applied to input image

    Returns:
        The image with size changes applied
        The new spacing for the resized image
    """

    # Iterate over list of operations:
    for i in size_changes:
        # the operation name is the first element, parameters are in second element
        op_to_perform = i[0]
        params = i[1]
        # if the operation is to resize to specified x, y dimension
        if op_to_perform == size_change_resize_to_x_y:
            new_size_0 = params[2]
            new_size_1 = params[3]
            img_np, spacing, _ = resize_to_x_y(img_np, spacing, new_size_0, new_size_1, anti_aliasing=anti_aliasing,
                                               interp_order=interp_order)

        # if the operation is to pad a specified axis with constants
        elif op_to_perform == size_change_pad_axis_with_total:
            axis_to_pad = params[0]
            total_pad = params[1]
            pad_value = params[2]
            img_np, _ = pad_axis_with_total(img_np, axis_to_pad, total_pad, pad_value)

        # if the operation is to crop using specified x and y values
        elif op_to_perform == size_change_crop_with_params:
            # orig_size_x = params[0]
            # orig_size_y = params[1]
            minx = params[2]
            maxx = params[3]
            miny = params[4]
            maxy = params[5]
            img_np, _ = crop_with_params(img_np, [minx, maxx, miny, maxy])

        else:
            print('ERROR, the operation', opt_to_perform, 'is not implemented in apply_size_changes_to_img')
            return None

    return img_np, spacing


def reverse_size_changes_to_img(img_np, spacing, size_changes, anti_aliasing=True, interp_order=1):
    """
    Reverses specified size changes on the given image.  Algorithms which change size (e.g. cxr standardization) can return size_changes information in list format
    This method is a utility to allow users to reverse those size changes at a later stage on any image
    Args:
        img_np: The image to be processed, xy ordering
        size_changes: The list of size changes provided by an algorithm which implemented them

    Returns:
        out_img_np: A resized version of the input image, where the specified size changes have been reversed
        spacing: the new spacing for the resized image

    """
    # Iterate over list in reverse order of operations:
    for i in reversed(size_changes):
        # the operation name is the first element, parameters are in second element
        op_to_perform = i[0]
        params = i[1]
        # if the operation performed was a resize to specific x, y dimensions
        if op_to_perform == size_change_resize_to_x_y:
            # the original size is stored in params 0 and 1
            return_to_size_0 = params[0]
            return_to_size_1 = params[1]
            # return to original size
            img_np, spacing, _ = resize_to_x_y(img_np, spacing, return_to_size_0, return_to_size_1,
                                               anti_aliasing=anti_aliasing, interp_order=interp_order)

        # if the operation performed was a padding of a specified axis with constants
        elif op_to_perform == size_change_pad_axis_with_total:
            axis_was_padded = params[0]
            total_pad = params[1]
            # undo the padding operation
            img_np, _ = un_pad_axis_with_total(img_np, axis_was_padded, total_pad)

        # if the operation performed was a crop using specifed x, y parameters
        elif op_to_perform == size_change_crop_with_params:
            # original size is stored in params 0 and 1
            orig_size_x = params[0]
            orig_size_y = params[1]
            minx = params[2]
            maxx = params[3]
            miny = params[4]
            maxy = params[5]
            # undo the crop operation
            img_np, _ = uncrop_with_params(img_np, orig_size_x, orig_size_y, [minx, maxx, miny, maxy])

        else:
            print('ERROR, the operation', op_to_perform, 'is not implemented in reverse_size_changes_to_img')
            return None

    return img_np, spacing
