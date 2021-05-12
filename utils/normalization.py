
import numpy as np
import resize_rescale
from scipy import ndimage
import mask_crop

# Keelin: I implemented this as per Rick's code at some point, but I don't really think it is needed.
def fix_scanline(img_in):
    x_max = img_in.shape[0]
    y_max = img_in.shape[1]

    # 2nd gaussian deriv with sigma at 0.7
    ridges = ndimage.filters.gaussian_filter(img_in, sigma=0.7, order=2, mode='wrap')

    # project
    projection = np.sum(ridges, axis=1)
    print('xmax', x_max, 'proj shape', projection.shape)

    # determine cut off point
    # remove 1% highest values
    tmp = np.sort(projection)
    n = 0.01 * len(tmp)
    print(n)
    without_outliers = tmp[0:int(np.round(len(tmp)-n))]
    print(len(without_outliers))
    # get statistics
    mean = np.mean(without_outliers)
    std = np.std(without_outliers)

    # determine standard scores
    max_score = 0
    max_x = 0
    i_x = int(0.05*x_max)
    while i_x < int(0.95*x_max):
        if std == 0:
            score = 1.01 * max_score
        else:
            score = (projection[i_x] - mean)/std

        if score > max_score:
            max_score = score
            max_x = i_x
        i_x += 1

    median_half_length = 5
    replace_x_start = max_x - 3
    replace_x_end = replace_x_start + 2 * 3 + 1

    num_to_replace = replace_x_end - replace_x_start
    for y in np.arange(0, y_max):
        start_val = img_in[replace_x_start - 1, y]
        end_val = img_in[replace_x_end, y]
        incr = float(end_val - start_val)/num_to_replace
        for x in np.arange(replace_x_start, replace_x_end):
            img_in[x,y] = start_val + incr * (x-replace_x_start + 1)

    return img_in

def split_energy_bands(image, sigmas):
    # We have one extra energy band which stores the final filtered image
    # this is not written in Rick's paper but is written in his reference [40]
    bands = np.zeros((len(sigmas)+1,) + image.shape)

    curr_I_x = image
    for sigma_ind, sigma in enumerate(sigmas):
        curr_L_x = ndimage.filters.gaussian_filter(curr_I_x, sigma, mode='wrap')

        bands[sigma_ind] = curr_I_x - curr_L_x

        # when we have no sigmas left, store the final filtered image in the last band (Rick's ref [40])
        if sigma_ind == len(sigmas) - 1:
            bands[sigma_ind + 1] = curr_L_x

        curr_I_x = curr_L_x  # this is not how Rick writes it in formula 2 but it IS how it is written in his reference [40] (and in his code!)

    # return the bands in reverse order of sigmas ([16,8,4,2,1])
    return bands[::-1]

# given the energy bands of the image, and a region to focus on
# returns the means, stdevs of the energy bands in that region
# the stdevs are the "energy values" e_i_omega
def report_energy_bands(bands, mask=1):
    means = []
    stdevs = []
    shape = bands[0].shape
    if isinstance(mask, np.ndarray):
        for band in bands:
            values = band[mask > 0]
            means.append(values.mean())
            stdevs.append(values.std())
    else:
        for band in bands:
            sub_im = band[int(0.15 * shape[0]):int(0.85*shape[0]), int(0.15 * shape[1]):int(0.85*shape[1])]
            means.append(sub_im.mean())
            stdevs.append(sub_im.std())
    return means, stdevs


def reconstruct(bands, means, stdevs, coefficients, lung_mean=None, mediastinum_mean=None):
    # Keelin: I do not see this step mentioned in Rick's paper....
    # but it seems to make sense and it doesn't work well without it!!
    bands[0] = (bands[0] - means[0])/stdevs[0]

    # ANNET: Here we check if the scaling between lung and mediastinum is okay, if not we adjust a bit
    # Keelin, leaving this part out for now as it is not in Rick's reference paper that we refer to.... but note
    # this may be needed in future
    """
    if lung_mean and mediastinum_mean:
        factor = 3 / (mediastinum_mean - lung_mean)
        if factor > 0:
            bands[0] = bands[0] - lung_mean
            bands[0] = bands[0] * factor
    """
    for j in range(1, bands.shape[0]):
        #print('band', j, 'coeff', coefficients[j], 'stdev', stdevs[j])
        bands[j] = bands[j] * coefficients[j]/stdevs[j]
    return bands.sum(0)


def report_energy_lungs_and_mediastinum(band, mean, stdev, mask):
    # ANNET: You did not have this function, it is used to do some scaling between the values in the
    # lungs and the values in the mediastinum. In most cases it doesn't change the outcome a lot,
    # but it can make small differences

    # Keelin, not calling this for now, as we stick to Rick's reference paper but note it could be used in the future

    band = band - mean
    band = band / stdev
    lung_values = []
    mediastinum_values = []
    for y in range(mask.shape[0]):
        first = -1
        last = -1
        for x in range(mask.shape[1]):
            if mask[y, x] > 0:
                if first == -1:
                    first = x
                last = x
        if first > -1 and last > -1:
            for x in range(first, last + 1):
                if mask[y, x] == 0:
                    mediastinum_values.append(band[y, x])
                elif mask[y, x] > 0:
                    lung_values.append(band[y, x])
    lung_values = sorted(lung_values)
    mediastinum_values = sorted(mediastinum_values)

    lung_fraction = round(len(lung_values) * 0.05)
    mediastinum_fraction = round(len(mediastinum_values) * 0.05)
    valid_lung_values = lung_values[lung_fraction:-lung_fraction]
    valid_mediastinum_values = mediastinum_values[mediastinum_fraction:-mediastinum_fraction]

    lung_mean = sum(valid_lung_values) / len(valid_lung_values)
    mediastinum_mean = sum(valid_mediastinum_values) / len(valid_mediastinum_values)

    return lung_mean, mediastinum_mean

def get_norm_central_70(img_np, spacing):
    """
    param: img_np - the input image
    Uses Rick's method to get normalization based on central 70% of the image.
    Will first crop the image and resize it to 2048 wide (preserving aspect ratio)
    :return: two versions of the norm image are returned.  One is the version for feeding to step 2 of the
    normalization algorithm, the other is a cleaned up (rescaled/clipped) version which is nicer for actually looking at.
    It also returns a list of the crop/resize operations that were carried out so that any related (e.g mask) images
    can be altered in the same way.
    """
    # the coefficients lambda based on a reference set
    rick_coeffs70 = [1, 0.142, 0.096, 0.070, 0.051, 0.070]
    sigmas = sigmas = [1, 2, 4, 8, 16]
    # make sure the image is float
    img = img_np.astype(np.float64)
    # crop away black borders
    img_np, crop_params = mask_crop.crop_img_borders(img_np, in_thresh_factor=0.05)
    # resize to width of 2048
    img_np, new_spacing = resize_rescale.resize_preserve_aspect_ratio(img_np, spacing, 2048, 0)
    # split into energy bands
    bands = split_energy_bands(img_np, sigmas)
    # get means and stddevs of the energy bands
    means, stdevs = report_energy_bands(bands)
    # reconstruct an image from the energy bands and the reference values
    norm_70 = reconstruct(bands, means, stdevs, rick_coeffs70)

    # the norm_70 image can be fed to the next normalization step.  But for a nice readable image we can scale/clip it.
    # set to min, max 0,4095
    new_min = 0
    new_max = 4095
    img_mean = norm_70.mean()
    set_min = img_mean - 5.0
    set_max = img_mean + 5.0
    readable_img = np.clip((new_max - new_min) * ((norm_70-set_min)/(set_max - set_min)) + new_min, new_min, new_max).astype(np.uint16)

    # Finally want to return a list of image size alterations we made:
    size_changes = {'crop_minxmaxx_minymaxy': crop_params, 'resize_preserve_aspect_ratio': [2048, 0]}

    # return the norm_70 for step2, the readable norm70, the new spacing, and the list of size changes
    return norm_70, readable_img, new_spacing, size_changes



def get_norm_lung_mask(img_np_norm70, lung_mask_np):
    """
    The second normalization step.  This takes the norm_70 image from get_norm_central_70, and a lung mask of it.
    The second normalization step is applied.
    :param img_np:
    :param spacing:
    :param lung_mask_np:
    :return: the final normalized image.
    """
    # the coefficients lambda based on a reference set
    rick_coeffs_lungseg = [1, 0.235, 0.169, 0.127, 0.093, 0.113]
    sigmas = [1, 2, 4, 8, 16]
    # split into energy bands
    bands = split_energy_bands(img_np_norm70, sigmas)
    # get means and stddevs of the energy bands
    means, stdevs = report_energy_bands(bands, mask=lung_mask_np)

    # for now we omit this step as it is not part of Rick's paper, but could help in the future
    # lung_mean, mediastinum_mean = report_energy_lungs_and_mediastinum(bands[0], means[0], stdevs[0], lungseg)

    norm = reconstruct(bands, means, stdevs, rick_coeffs_lungseg)#omit for now:, lung_mean, mediastinum_mean)

    # rescale and clip to more readable values:
    new_min = 0
    new_max = 4095
    #
    set_min = - 5.0
    set_max = 5.0

    # rescale from range -5, +5 to 0,4095
    # and clip values that end up outside that range
    readable_img = np.clip((new_max - new_min) * ((norm-set_min)/(set_max - set_min)) + new_min, new_min, new_max)

    return readable_img
