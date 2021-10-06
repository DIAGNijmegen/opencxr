# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:40:35 2020
@author: keelin
"""

from pathlib import Path

import opencxr
from opencxr.utils.file_io import read_file

# Load the algorithm
img_sorter_algorithm = opencxr.load(opencxr.algorithms.image_sorter)
# read in a test image
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
#   'Type_Probs_PA_AP_lateral_not-CXR': array([[9.9999976e-01, 2.5101654e-08, 2.4382584e-07, 1.0590604e-08]],dtype=float32),
#   'Rotation_Probs_0_90_180_270': array([[9.9999988e-01, 2.7740466e-08, 2.2800064e-08, 3.7591672e-08]],dtype=float32),
#   'Inversion_Probs_No_Yes': [array([[0.99999684]], dtype=float32), array([[3.1410489e-06]], dtype=float32)],
#   'Lateral_Flip_No_Yes': [array([[0.9999987]], dtype=float32), array([[1.324667e-06]], dtype=float32)]}

# Possible values for the first 4 keys listed here are as follows:
#       Type:  ['PA', 'AP', 'lateral', 'not-CXR']
#       Rotation: ['0', '90', '180', '270']
#       Inversion: ['No', 'Yes']
#       Lateral_Flip: ['No', 'Yes']
result = img_sorter_algorithm.run(img_np)
print(result)
