# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:40:35 2020
@author: keelin
"""

from pathlib import Path
import numpy as np
import opencxr
from opencxr.utils.file_io import read_file


def run_imagesorter():
    # Load the algorithm
    img_sorter_algorithm = opencxr.load(opencxr.algorithms.image_sorter)
    # read in a test image
    # note that the image sorter is designed to run on raw data so it performs better without prior image normalization/standardization
    f_in = Path(__file__).parent / "resources" / "images" / "c0004.mha"
    f_in = str(f_in.resolve())
    img_np, spacing, pydata = read_file(f_in)
    # get the output from the image sorter
    # the output is a dict something like the one shown below.
    # The first four keys give classifications for Type, Rotation, Inversion, Lateral_Flip.
    # The second four keys provide probabilities of all possible classes for users that might need this
    # {'Type': 'PA',
    #  'Rotation': '0',
    #   'Inversion': 'No',
    #   'Lateral_Flip': 'No',
    #   'Type_Probs_PA_AP_lateral_notCXR': [0.99999976, 2.5101654e-08, 2.4382584e-07, 1.0590604e-08],
    #   'Rotation_Probs_0_90_180_270': [0.9999999, 2.7740466e-08, 2.2800064e-08, 3.7591672e-08],
    #   'Inversion_Probs_No_Yes': [0.9999968589511354, 3.1410489e-06],
    #   'Lateral_Flip_Probs_No_Yes': [0.9999986753330177, 1.324667e-06]}

    # Possible values for the first 4 keys listed here are as follows:
    #       Type:  ['PA', 'AP', 'lateral', 'notCXR']
    #       Rotation: ['0', '90', '180', '270']
    #       Inversion: ['No', 'Yes']
    #       Lateral_Flip: ['No', 'Yes']
    result = img_sorter_algorithm.run(img_np)

    expected_result = {
        "Type": "PA",
        "Rotation": "0",
        "Inversion": "No",
        "Lateral_Flip": "No",
        "Type_Probs_PA_AP_lateral_notCXR": [
            0.99999976,
            2.5101654e-08,
            2.4382584e-07,
            1.0590604e-08,
        ],
        "Rotation_Probs_0_90_180_270": [
            0.9999999,
            2.7740466e-08,
            2.2800064e-08,
            3.7591672e-08,
        ],
        "Inversion_Probs_No_Yes": [0.9999968589511354, 3.1410489e-06],
        "Lateral_Flip_Probs_No_Yes": [0.9999986753330177, 1.324667e-06],
    }

    first_four_keys_passed = False
    if (
        result["Type"] == expected_result["Type"]
        and result["Rotation"] == expected_result["Rotation"]
        and result["Inversion"] == expected_result["Inversion"]
        and result["Lateral_Flip"] == expected_result["Lateral_Flip"]
    ):
        first_four_keys_passed = True
    else:
        print("Failed to match first four keys checked", result, expected_result)

    last_four_keys_passed = False

    if (
        np.array_equal(
            np.round(np.asarray(result["Type_Probs_PA_AP_lateral_notCXR"]), 3),
            np.round(np.asarray(expected_result["Type_Probs_PA_AP_lateral_notCXR"]), 3),
        )
        and np.array_equal(
            np.round(np.asarray(result["Rotation_Probs_0_90_180_270"]), 3),
            np.round(np.asarray(expected_result["Rotation_Probs_0_90_180_270"]), 3),
        )
        and np.array_equal(
            np.round(np.asarray(result["Inversion_Probs_No_Yes"]), 3),
            np.round(np.asarray(expected_result["Inversion_Probs_No_Yes"]), 3),
        )
        and np.array_equal(
            np.round(np.asarray(result["Lateral_Flip_Probs_No_Yes"]), 3),
            np.round(np.asarray(expected_result["Lateral_Flip_Probs_No_Yes"]), 3),
        )
    ):
        last_four_keys_passed = True
    else:
        print("Failed to match last four keys checked", result, expected_result)

    if first_four_keys_passed and last_four_keys_passed:
        print("Image Sorter test completed successfully")
        return 1
    else:
        print("Image Sorter results not as expected")
        return 0


def test_imagesorter():
    assert run_imagesorter() == 1


if __name__ == "__main__":
    run_imagesorter()
