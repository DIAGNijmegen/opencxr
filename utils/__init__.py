
import opencxr
from opencxr.utils.resize_rescale import resize_to_x_y, un_pad_axis_with_total, pad_axis_with_total
from opencxr.utils.mask_crop import crop_with_params, uncrop_with_params

size_change_resize_to_x_y = 'sc_resize_to_x_y'
size_change_pad_axis_with_total = 'sc_pad_axis_with_total'
size_change_unpad_axis_with_total = 'sc_unpad_axis_with_total'
size_change_crop_with_params = 'sc_crop_with_params'
size_change_uncrop_with_params = 'sc_uncrop_with_params'

def apply_size_changes_to_img(img_np, spacing, size_changes, anti_aliasing=True, interp_order=1):
    """
    Applies specified size changes to an image.  Methods which change size will return size_changes information in list format
    This method is a utility to allow users to apply those same size changes to another image.
    Args:
        img_np:
        size_changes:

    Returns:

    """
    print('size changes are', size_changes)
    # Iterate over list of operations:
    for i in size_changes:
        print(i)
        op_to_perform = i[0]
        params = i[1]
        if op_to_perform == size_change_resize_to_x_y:
            new_size_0 = params[2]
            new_size_1 = params[3]
            img_np, spacing, _ = resize_to_x_y(img_np, spacing, new_size_0, new_size_1, anti_aliasing=anti_aliasing, interp_order=interp_order)

        elif op_to_perform == size_change_pad_axis_with_total:
            axis_to_pad = params[0]
            total_pad = params[1]
            pad_value = params[2]
            img_np, _ = pad_axis_with_total(img_np, axis_to_pad, total_pad, pad_value)

        elif op_to_perform == size_change_crop_with_params:
            #orig_size_x = params[0]
            #orig_size_y = params[1]
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
    Reverses specified size changes on the given image.  Methods which change size will return size_changes information in list format
    This method is a utility to allow users to reverse those size changes at a later stage on any image
    Args:
        img_np: The image to be processed, xy ordering
        size_changes: The list of size changes provided by resize methods

    Returns:
        out_img_np: An image with the size changes reversed

    """
    # Iterate over list in reverse order of operations:
    for i in reversed(size_changes):
        print(i)
        op_to_perform = i[0]
        params = i[1]
        if op_to_perform == size_change_resize_to_x_y:
            return_to_size_0 = params[0]
            return_to_size_1 = params[1]
            img_np, spacing, _ = resize_to_x_y(img_np, spacing, return_to_size_0, return_to_size_1, anti_aliasing=anti_aliasing, interp_order=interp_order)

        elif op_to_perform == size_change_pad_axis_with_total:
            axis_was_padded = params[0]
            total_pad = params[1]
            img_np, _ = un_pad_axis_with_total(img_np, axis_was_padded, total_pad)

        elif op_to_perform == size_change_crop_with_params:
            orig_size_x = params[0]
            orig_size_y = params[1]
            minx = params[2]
            maxx = params[3]
            miny = params[4]
            maxy = params[5]
            img_np, _ = uncrop_with_params(img_np, orig_size_x, orig_size_y, [minx, maxx, miny, maxy])

        else:
            print('ERROR, the operation', op_to_perform, 'is not implemented in reverse_size_changes_to_img')
            return None

    return img_np, spacing
