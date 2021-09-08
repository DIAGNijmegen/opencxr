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
# the output is a dict something like: {'Type': 'PA', 'Rotation': '0', 'Inversion': 'No', 'Lateral Flip': 'No'}
# The keys are always the same.  Possible values for the keys are as follows:
#       Type:  ['PA', 'AP', 'lateral', 'not-CXR']
#       Rotation: ['0', '90', '180', '270']
#       Inversion: ['No', 'Yes']
#       Lateral Flip: ['No', 'Yes']
result = img_sorter_algorithm.run(img_np)
print(result)
