# -*- coding: utf-8 -*-
"""

@author: keelin
"""
import os.path
from pathlib import Path
from opencxr.utils.file_io import read_file, write_file
from opencxr.utils.mask_crop import (
    crop_img_borders,
    crop_img_borders_by_edginess,
    uncrop_with_params,
    set_non_mask_constant,
)
from opencxr.utils.rotate_flip_invert import (
    rotate_img,
    flip_x,
    flip_y,
    invert_grayscale,
)
from opencxr.utils import reverse_size_changes_to_img
import numpy as np


def test_file_io():
    # read files of type dicom, png (8 and 16 bit), mha
    f_in_dcm = Path(__file__).parent / "resources" / "images" / "c0004.dcm"
    f_in_dcm = str(f_in_dcm.resolve())
    img_np_dcm, spacing_dcm, tags = read_file(f_in_dcm)

    f_in_png_8 = Path(__file__).parent / "resources" / "images" / "c0001_8bit.png"
    f_in_png_8 = str(f_in_png_8.resolve())
    img_np_png_8, _, _ = read_file(f_in_png_8)

    f_in_png_16 = Path(__file__).parent / "resources" / "images" / "c0002_16bit.png"
    f_in_png_16 = str(f_in_png_16.resolve())
    img_np_png_16, _, _ = read_file(f_in_png_16)

    f_in_mha = Path(__file__).parent / "resources" / "images" / "c0003.mha"
    f_in_mha = str(f_in_mha.resolve())
    img_np_mha, spacing_mha, _ = read_file(f_in_mha)

    # verify the appropriate information is not empty
    passing = True
    if not img_np_dcm.shape == (2992, 2991):
        print("dcm wrong shape", img_np_dcm.shape)
        passing = False
    if not np.array_equal(spacing_dcm, np.asarray([0.143, 0.143, 1.0])):
        print("dcm wrong spacing", spacing_dcm)
        passing = False
    if (
        not tags["StudyInstanceUID"].value
        == "9999.324256108382831452380358921215687044879"
    ):
        print("dcm tags wrong study UID", tags["StudyInstanceUID"].value)
        passing = False

    if not img_np_png_8.shape == (1024, 1024):
        print("png 8 bit wrong shape", img_np_png_8.shape)
        passing = False

    if not img_np_png_16.shape == (3000, 2967):
        print("png 16 bit wrong shape", img_np_png_16.shape)
        passing = False

    if not img_np_mha.shape == (2834, 2851):
        print("mha wrong shape", img_np_mha.shape)
        passing = False
    if not np.array_equal(spacing_mha, np.asarray([0.148, 0.148])):
        print("mha wrong spacing", spacing_mha)
        passing = False

    # write files of type mha, png (uint8) and png (uint16)
    f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "writing_mha.mha"
    f_out = str(f_out.resolve())
    write_file(f_out, img_np_mha, spacing_mha)
    if not os.path.isfile(f_out):
        print("Failed to write file at ", f_out)
        passing = False

    f_out = (
        Path(__file__).parent
        / "resources"
        / "tmp_test_outputs"
        / "writing_png_8bit.png"
    )
    f_out = str(f_out.resolve())
    write_file(f_out, img_np_png_8, [1.0, 1.0])
    if not os.path.isfile(f_out):
        print("Failed to write file at ", f_out)
        passing = False

    f_out = (
        Path(__file__).parent
        / "resources"
        / "tmp_test_outputs"
        / "writing_png_16bit.png"
    )
    f_out = str(f_out.resolve())
    write_file(f_out, img_np_png_16, [1.0, 1.0])
    if not os.path.isfile(f_out):
        print("Failed to write file at ", f_out)
        passing = False

    assert passing


def test_mask_crop():
    passing = True
    # read in an image with a black border present
    f_in = Path(__file__).parent / "resources" / "images" / "c3574.png"
    f_in = str(f_in.resolve())
    img_np, _, _ = read_file(f_in)

    # test first crop_img method
    cropped_img, size_changes = crop_img_borders(img_np)
    if not cropped_img.shape == (1024, 984):
        print("crop_img_borders returned wrong shape", cropped_img.shape)
        passing = False

    # test second crop_img method
    cropped_img, size_changes = crop_img_borders_by_edginess(img_np)
    if not cropped_img.shape == (1022, 984):
        passing = False

    # test uncrop with params method
    img_np_uncropped, _ = reverse_size_changes_to_img(
        cropped_img, [1.0, 1.0], size_changes
    )

    if not img_np_uncropped.shape == (1024, 1024):
        print("uncropped shape not as expected", img_np_uncropped.shape)
        passing = False

    # test set non-mask as constant
    mask_np = np.zeros(img_np.shape)
    mask_np[500:600, 500:600] = 1

    masked_img = set_non_mask_constant(img_np, mask_np)

    # verify masked image is correct size
    if not masked_img.shape == img_np.shape:
        print("masked image has wrong shape", masked_img.shape)
        passing = False
    # verify pixels we reset are zero now and were not in original
    test_patch_masked = masked_img[0:450, 0:450]
    test_patch_unmasked = img_np[0:450, 0:450]
    if not np.all(test_patch_masked == 0):
        print("mask not applied correctly")
        passing = False
    if np.all(test_patch_unmasked == 0):
        print("orig image was already zero in test patch")
        passing = False

    assert passing


def test_rotate_flip_invert():
    passing = True
    f_in = Path(__file__).parent / "resources" / "images" / "c0005.mha"
    f_in = str(f_in.resolve())
    img_np, spacing, _ = read_file(f_in)

    rot90 = rotate_img(img_np, 90)
    rot180 = rotate_img(img_np, 180)
    rot270 = rotate_img(img_np, 270)
    rot360 = rotate_img(img_np, 360)

    flipyx = flip_y(flip_x(img_np))

    flipxyrot90 = flip_x(flip_y(rot90))

    # compare images we expect to be the same
    diff_img = img_np.astype(float) - rot360.astype(float)
    avg_pixel_diff = np.sum(np.abs(diff_img)) / (img_np.shape[0] * img_np.shape[1])
    if not avg_pixel_diff < 20:
        print("rot360 not the same as orig", avg_pixel_diff)
        passing = False

    diff_img = flipyx.astype(float) - rot180.astype(float)
    avg_pixel_diff = np.sum(np.abs(diff_img)) / (img_np.shape[0] * img_np.shape[1])
    if not avg_pixel_diff < 20:
        print("flipyx not same as rot180", avg_pixel_diff)
        passing = False

    diff_img = flipxyrot90.astype(float) - rot270.astype(float)
    avg_pixel_diff = np.sum(np.abs(diff_img)) / (img_np.shape[0] * img_np.shape[1])
    if not avg_pixel_diff < 20:
        print("flipxyrot90 not same as rot270", avg_pixel_diff)
        passing = False

    assert passing


if __name__ == "__main__":
    test_file_io()
    test_mask_crop()
    test_rotate_flip_invert()
