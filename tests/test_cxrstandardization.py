# -*- coding: utf-8 -*-
"""

@author: keelin
"""

from pathlib import Path
import opencxr
from opencxr.utils import apply_size_changes_to_img
from opencxr.utils.file_io import read_file, write_file
import numpy as np


def run_cxrstandardization():
    # read in and standardize an image
    # first get the file path relative to current location
    f_in = Path(__file__).parent / "resources" / "images" / "c0002.mha"
    f_in = str(f_in.resolve())
    # read in the image
    img_np, spacing, pydata = read_file(f_in)
    # load the standardization algorithm
    cxrstandardize_algorithm = opencxr.load(opencxr.algorithms.cxr_standardize)

    # run the standardization algorithm on the test image
    # this will return the new image, the new spacing, and a dict of size changes carried out
    # these size changes can be easily applied to other images, see further code below
    final_norm_img, new_spacing, size_changes = cxrstandardize_algorithm.run(
        img_np, spacing
    )

    # set up an output location and write the output image there
    f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "c0002_norm.mha"
    f_out = str(f_out.resolve())
    write_file(f_out, final_norm_img, new_spacing)

    # Next we test whether the size_changes information that was provided is correct
    # run apply_size_changes_to_img and verify that the output sizes are the same as the ones that came from standardization algorithm
    img_resized_to_test, new_spacing_to_test = apply_size_changes_to_img(
        img_np, spacing, size_changes
    )

    resized_img_worked = final_norm_img.shape == img_resized_to_test.shape
    resized_spacing_worked = new_spacing == new_spacing_to_test

    if (
        final_norm_img.shape == (1024, 1024)
        and not np.max(final_norm_img) == 0
        and (final_norm_img.shape == img_resized_to_test.shape)
        and (new_spacing == new_spacing_to_test)
    ):
        print("CXR standardization test completed successfully")
        return 1
    else:
        print("CXR standardization results not as expected")
        return 0


def test_cxrstandardization():
    assert run_cxrstandardization() == 1


if __name__ == "__main__":
    run_cxrstandardization()
