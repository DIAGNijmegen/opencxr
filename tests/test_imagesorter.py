# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:40:35 2020
@author: keelin
"""

import opencxr
from pathlib import Path
from opencxr.utils.file_io import read_file
# Load the algorithm
# possible algorithms are listed in opencxr/algorithms/__init__.py
img_sorter_algorithm = opencxr.load(opencxr.algorithms.image_sorter)

f_in = Path(__file__).parent / "resources" / "images" / "g0019.mha"
f_in = str(f_in.resolve())
img_np,spacing,pydata = read_file(f_in)

result = img_sorter_algorithm.run(img_np)
print(result)
