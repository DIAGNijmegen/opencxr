# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:40:35 2020
@author: keelin
"""

import opencxr
from pathlib import Path

# Load the algorithm
# possible algorithms are listed in opencxr/algorithms/__init__.py
img_sorter_algorithm = opencxr.load(opencxr.algorithms.image_sorter)

# test with input and output folders
f_in = Path(__file__).parent / "resources" / "images"
f_out = Path(__file__).parent / "resources" / "images"
img_sorter_algorithm.run(f_in, f_out)

# TODO: test with input and output files