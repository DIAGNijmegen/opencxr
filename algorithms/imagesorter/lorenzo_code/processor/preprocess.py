'''
Author : Lorenzo Valacchi (lorenzovalac@gmail.com)

Preprocessing functionalities.
'''

from __future__ import division
import numpy as np
import os
from skimage import color
from PIL import Image
from opencxr.utils.file_io import read_dicom, read_mhd_mha, read_png, read_file
from opencxr.utils.mask_crop import crop_img_borders
from opencxr.utils.resize_rescale import rescale_to_min_max, resize_isotropic
import SimpleITK as sitk
import timeit



def get_image_list(input_folder):
    """
    Inspect the given input folder and return a list of processable images.

    good formats: .png, .mha, .mhd, .dcm

    Return:
    path_list :  list of processable images.
    name_list : list of corresponding names
    ext_list : list of corresponding extensions
    
    """

    path_list = []
    name_list = []
    ext_list = []
    print('traversing folders to find all image files.......')
    for r, d, f in os.walk(input_folder):
        
        for file in f:
            # print('considering file', file)
            name, ext = os.path.splitext(file)
            ext = ext.replace(".", "")
            # print('name and ext are ', name, ext)

            if (ext.lower() == 'png') or (ext.lower() == 'mha') or (ext.lower() == 'dcm') or (ext.lower() == 'dc3'):
                path = os.path.join(r,file)
                path_list.append(path)
                name_list.append(name)
                ext_list.append(ext)
                
            elif ext.lower() == 'mhd':
                #check if .zraw file is present in the same folder
                if os.path.isfile(os.path.join(r,name+'.zraw')) or os.path.isfile(os.path.join(r,name+'.raw'))  or os.path.isfile(os.path.join(r,name+'.ZRAW')) or os.path.isfile(os.path.join(r,name+'.RAW')):
                    path = os.path.join(r,file)
                    path_list.append(path)
                    name_list.append(name)
                    ext_list.append(ext)
                    
                else:
                    print('will not process file {}: mhd without zraw or raw in same folder'.format(file))
        
    return path_list, name_list, ext_list

def clip_at_percentiles(img_np, perc_low, perc_high):
    in_dtype = img_np.dtype
    low_clip = np.percentile(img_np, perc_low)
    high_clip = np.percentile(img_np, perc_high)
    return np.clip(img_np, low_clip, high_clip).astype(in_dtype)


def resize_and_pad(img_np, spacing):
    """
    Resizes such that the larger dimension goes to 256 and spacing stays isotropic
    Then pad such that the smaller dimension is also 256
    """
    # pad with 0 (data should be non-inverted by now)
    pad_value = 0
    
    shape_x, shape_y = img_np.shape
    
    if shape_x >= shape_y:
        # do resize
        img_np_256, new_spacing = resize_isotropic(img_np, [spacing[0], spacing[1]], 256, axis_specified=0, anti_aliasing=True, interp_order=1)
        
        # pad smaller dimension (in this case top and bottom)
        diff = 256 - img_np_256.shape[1]
        top_pad = int(np.round(float(diff)/2))
        bottom_pad = 256 - img_np_256.shape[1] - top_pad
        img_np_256 = np.pad(img_np_256, ((0,0),(top_pad,bottom_pad)), 'constant', constant_values=pad_value)
        
    else:
        # do resize
        img_np_256, new_spacing = resize_isotropic(img_np, [spacing[0], spacing[1]], 256, axis_specified=1, anti_aliasing=True, interp_order=1)
        
        # pad smaller dimension (in this case left and right)
        diff = 256 - img_np_256.shape[0]
        left_pad = int(np.round(float(diff)/2))
        right_pad = 256 - img_np_256.shape[0] - left_pad
        img_np_256 = np.pad(img_np_256, ((left_pad,right_pad), (0,0)), 'constant', constant_values=pad_value)
        
    return img_np_256, [new_spacing, new_spacing]


def preprocess_images(path_list, ext_list):
    """
    Preprocess a list of images.

    Preprocessing pipeline:
    - Homogeneous border removal around the image
    - isotropical resize of the bigger dimension to 256
    - padding of teh lower dimension to 256
    - convert the grayscale image to rgb
    - normalize the image

    Return:
    X :  matrix of preprocessed images
    
    """
    
    # initialize vector to save processed images
    #km X = np.empty((len(path_list), 256, 256, 3))
    X = np.empty((len(path_list), 256, 256), dtype=np.uint8)
    
    
    num_imgs = len(path_list)
    for n, im in enumerate(path_list):
        print('about to preprocess on image ', n, 'of ', num_imgs, '(',  im, ')')
        start_time = timeit.default_timer()
        try:
            if ext_list[n].lower() == 'png':
                img_x_y = read_png(im)
                spacing_x_y = [1.0, 1.0]

            elif ext_list[n].lower() == 'dcm' or ext_list[n].lower() == 'dc3':
                img_x_y, spacing_x_y, pydicom_version = read_dicom(im)

            else:
                img_x_y, spacing_x_y = read_mhd_mha(im)
            print('reading image took', timeit.default_timer() - start_time)
        except Exception as e:
            print('skipping, unable to read image ', im)
            print(e)
            continue


        #Because sometimes images apparently have a 3rd dimension eg Y:\cxr\archives\RadboudCXR\1000/st000/se000\1.3.12.2.1107.5.3.49.22121.11.201410311125260796.dcm
        start_time = timeit.default_timer()
        img_x_y = np.squeeze(img_x_y)
        print('squeeze image took', timeit.default_timer() - start_time)

        
        start_time = timeit.default_timer()
        # check if the original image is rgb and convert to gray if so
        if len(img_x_y.shape)<2 or len(img_x_y.shape)>3:
            print('unable to proceed(1), ', im, ' img shape is ', img_x_y.shape)
            continue
        if len(img_x_y.shape) == 3:
            channel_axis = np.argmin(img_x_y.shape)
            # print('channel axis seems to be at ', channel_axis, img_x_y.shape)
            if channel_axis == 0: #swap axes as channel expected last
                img_x_y = np.swapaxes(img_x_y, 0, 2)
                img_x_y = np.swapaxes(img_x_y, 0, 1)
                # print('swapped axes, shape now is ', img_x_y.shape)
            if img_x_y.shape[2]==4:
                img_x_y = color.rgb2gray(color.rgba2rgb(img_x_y))
                # print('did 2 transforms, shape now is ', img_x_y.shape)
            elif img_x_y.shape[2]==3:
                img_x_y = color.rgb2gray(img_x_y)
                # print('did 1 transform shape now is ', img_x_y.shape)
            else:
                print('unable to proceed(2), ', im, ' img shape is ', img_x_y.shape)
                continue
            img_x_y = rescale_to_min_max(img_x_y, np.uint8, 0, 255)
        print('reformat image took', timeit.default_timer() - start_time)


        # crop any homogeneous border regions in the image
        start_time = timeit.default_timer()
        img_x_y, crop_values = crop_img_borders(img_x_y, in_thresh_factor=0.05)
        print('crop border image took', timeit.default_timer() - start_time)

        # clip
        start_time = timeit.default_timer()
        img_x_y = clip_at_percentiles(img_x_y, 1, 99)
        print('clip percentiles image took', timeit.default_timer() - start_time)


        # convert image to 8 bit unsigned
        #km if (img_x_y.dtype != 'uint8'):
        start_time = timeit.default_timer()
        img_x_y = rescale_to_min_max(img_x_y, new_dtype=np.uint8, new_min=0, new_max=255)
        print('rescale image took', timeit.default_timer() - start_time)


        # resize and pad with 0 to size 256, 256
        start_time = timeit.default_timer()
        img_x_y, ignore_new_spacing = resize_and_pad(img_x_y, spacing_x_y)
        print('resize and pad image took', timeit.default_timer() - start_time)

        
        start_time = timeit.default_timer()
        img_x_y = rescale_to_min_max(img_x_y, new_dtype=np.uint8, new_min=0, new_max=255)
        print('rescale image took', timeit.default_timer() - start_time)

        
        # print('img xy  type', img_x_y.dtype)

        # km img = np.transpose(img_x_y)
        # km img = Image.fromarray(img, 'P')
        # km img = np.asarray(img.convert('RGB'), dtype=np.uint8)
        start_time = timeit.default_timer()
        X[n] = np.transpose(img_x_y)
        print('transpose image took', timeit.default_timer() - start_time)

        # kmX[n,] = img/255.
    # print('type after preprocessing', X[3].dtype)
    # print('range after preprocessing', np.min(X[3]), np.max(X[3]))
    ####sitk.WriteImage(sitk.GetImageFromArray(X[3]), '/output/tempout.png')
    return X
