# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:58:39 2019
"""

import os

import SimpleITK as sitk
import numpy as np
import png  # pypng library to allow management of 16 bit png files
import pydicom
from skimage import color

"""
Methods to read and write CXR images of various formats
By convention we return (and expect) CXR images as numpy arrays where
shape[0] is the x-dimension (width) of the image
while shape[1] is they y-dimension (height)
"""


def read_file(in_img_location):
    """
    Wrapper method that checks which type of image it is and
    uses the appropriate method to read it. Supports mha, mhd, png, dicom
    :param in_img_location: The file location to be read
    :return: img_np: The image as an np array where shape[0] is the x-dimension (width) of the image
               while shape[1] is they y-dimension (height)
    """
    in_img_location = in_img_location.strip()
    if not os.path.isfile(in_img_location):
        raise Exception('Unable to find file at {}'.format(in_img_location))
    extension = os.path.splitext(in_img_location)[1]
    extension = extension.lower()
    if extension == '.mhd' or extension == '.mha':
        return read_mhd_mha(in_img_location)
    elif extension == '.dcm':
        return read_dicom(in_img_location)
    elif extension == '.png':
        return read_png(in_img_location)
    else:
        raise Exception('Do not recognize extension {} in location {}. \
        Unable to read file'.format(extension, in_img_location))


def read_png(in_img_location):
    """
    Method to read in a png image and return it as an np array
    :param in_img_location: The file location for the png image
    :return: The image as np array
             The spacing of the image
             An empty string as a place filler since no dicom tags available
    """
    # sitk now seems to handle png even better than pypng, which has some issues, so moving to sitk
    # weird images to debug with include chest-xray-14 image 00000317_000.png or /BIMCV-COVID19/sub-S03214/ses-E07979
    itk_img = sitk.ReadImage(in_img_location)
    init_np_array = np.squeeze(sitk.GetArrayFromImage(itk_img))

    # Can only handle input images with 2 or 3 axes
    if len(init_np_array.shape) < 2 or len(init_np_array.shape) > 3:
        print('unable to proceed(1), ', im, ' img shape is ', init_np_array.shape)
        return
    # If 3 axes then assume one of them is a channel axis, need to make sure it is in last position
    if len(init_np_array.shape) == 3:
        channel_axis = np.argmin(init_np_array.shape)
        # print('channel axis seems to be at ', channel_axis, init_np_array.shape)
        if channel_axis == 0:  # swap axes as channel expected last
            init_np_array = np.swapaxes(init_np_array, 0, 2)
            init_np_array = np.swapaxes(init_np_array, 0, 1)
            # print('swapped axes, shape now is ', init_np_array.shape)
        # if there are 4 channels then we assume we have rgba image.... make it grayscale
        if init_np_array.shape[2] == 4:
            init_np_array = color.rgb2gray(color.rgba2rgb(init_np_array))
            # print('did 2 transforms, shape now is ', init_np_array.shape)
        # if there are 3 channels then we assume we have rgb image.... make it grayscale
        elif init_np_array.shape[2] == 3:
            init_np_array = color.rgb2gray(init_np_array)
            # print('did 1 transform shape now is ', init_np_array.shape)
        else:
            print('unable to proceed(2), ', im, ' img shape is ', init_np_array.shape)
            return

    # return with x, y (width, height) axis ordering not numpy usual ordering
    img_np_x_y = np.transpose(init_np_array)
    spacing_x_y = np.transpose(itk_img.GetSpacing())

    return img_np_x_y, spacing_x_y, ''


def read_mhd_mha(in_img_location):
    """
    A method which reads in an mhd or mha image and returns it in
    X(axis 0), Y(axis 1) format with spacing
    :param in_img_location: The file location of the original image
    :return: The image as np array
             The spacing of the image
             An empty string as a place filler since no dicom tags available
    """
    itk_img = sitk.ReadImage(in_img_location)

    img_np_x_y = np.squeeze(np.transpose(sitk.GetArrayFromImage(itk_img)))
    spacing_x_y = np.transpose(itk_img.GetSpacing())

    return img_np_x_y, spacing_x_y, ''


def read_dicom(in_img_location):
    """
    Reads a dicom image
    :param in_img_location: The file location of the dicom image
    :return: The image as np array (x, y, width, height)
             The spacing of the image
             The dicom tags as provided by pydicom
    """
    """
    Reads dicom (twice).
    First using sitk - to get the pixel data.  This is reliable also if the
    data has been jpeg2000 compressed (where pydicom does not work)
    Secondly, using pydicom, to get the headers.  We explicitly remove the
    pydicom version of PixelData to avoid confusion.
    #TODO: Could use sitk to read dicom tag info?
    return np_img_array, spacing, pydicom data
    """
    itk_image = sitk.ReadImage(in_img_location)
    img_np_x_y = np.squeeze(np.transpose(sitk.GetArrayFromImage(itk_image)))
    spacing_x_y = np.transpose(itk_image.GetSpacing())
    # print('spacing from sitk image is', spacing_x_y)

    # Need this for headers, but pydicom cannot read pixel data if it is
    # stored in jpeg2000 compression
    pydicom_version = pydicom.dcmread(in_img_location)  #try this??? ds = pydicom.filereader.read_partial(f, stop_when=pixel_data_reached)

    # Make a correction because sitk seems to read pixel spacing only from the ImagerPixelSpacing dicom tag
    # In some images that tag is absent but the tag PixelSpacing is available
    if 'PixelSpacing' in pydicom_version and not 'ImagerPixelSpacing' in pydicom_version:
        # print('PixelSpacing dicom tag is ', pydicom_version.PixelSpacing)
        spacing_x_y = pydicom_version.PixelSpacing

    # Remove the pixel data from the pydicom data so only headers are returned
    del pydicom_version.PixelData
    return img_np_x_y, spacing_x_y, pydicom_version


def write_file(out_img_location, img_np_x_y, spacing_x_y=[1.0, 1.0]):
    """
    Wrapper method that checks file extension and uses the appropriate method to write it
    Note writing dicom is not supported
    :param out_img_location: The file location to write to (folders will be created if needed)
    :param img_np_x_y: The image to write in np array (x, y, width, height)
    :param spacing_x_y: The spacing of the image (if none then 1.0, 1.0 assumed)
    :return:
    """

    out_img_dir = os.path.dirname(out_img_location)
    os.makedirs(out_img_dir, exist_ok=True)

    extension = os.path.splitext(out_img_location)[1]
    if extension == '.mhd' or extension == '.mha':
        write_mhd_mha(out_img_location, img_np_x_y, spacing_x_y)
    elif extension == '.png':
        write_png(out_img_location, img_np_x_y)
    else:
        raise Exception('Do not recognize extension {} in location {},\
                     unable to write file'.format(extension, out_img_location))
    if not os.path.isfile(out_img_location):
        raise Exception('Failed to write image to location\
                                         {}'.format(out_img_location))


def write_png(out_img_location, img_np_x_y):
    """
    Write png image to disk.  Only uint16 or uint8 supported
    :param out_img_location: The file location to write to
    :param img_np_x_y: The np array of the image to write
    :return:
    """

    # still using pypng for this because sitk only supports writing uchar and ushort png images
    img_np_y_x = np.transpose(img_np_x_y)
    with open(out_img_location, 'wb') as f:
        if img_np_y_x.dtype == 'uint16':
            writer = png.Writer(width=img_np_y_x.shape[1],
                                height=img_np_y_x.shape[0],
                                greyscale=True,
                                bitdepth=16)
        elif img_np_y_x.dtype == 'uint8':
            writer = png.Writer(width=img_np_y_x.shape[1],
                                height=img_np_y_x.shape[0],
                                greyscale=True,
                                bitdepth=8)
        else:
            raise Exception('only 8 bit and 16 bit png writing supported.\
                            Please provide np array as uint16 or uint8. \
                            Failed to write {}'.format(out_img_location))
        img_as_list_rows = img_np_y_x.tolist()
        writer.write(f, img_as_list_rows)


def write_mhd_mha(out_img_location, img_np_x_y, spacing_x_y):
    """
    A method which writes an mhd or mha image
    given the np version in X(axis 0), Y(axis 1) format with spacing
    :param out_img_location: The file location to write to
    :param img_np_x_y: The np array in x, y, width, height format
    :param spacing_x_y: The spacing of the image
    :return:
    """
    itk_out = sitk.GetImageFromArray(np.transpose(img_np_x_y))
    itk_out.SetSpacing(np.transpose(spacing_x_y))

    sitk.WriteImage(itk_out, out_img_location, True)
