# -*- coding: utf-8 -*-
"""

@author: ecem
"""

import opencxr
from pathlib import Path
from opencxr.utils.file_io import read_file

# Load the algorithm
# possible algorithms are listed in opencxr/algorithms/__init__.py
lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)
# test with numpy array as input
f_in = Path(__file__).parent / "resources" / "images" / "test_img1.png" 
img,spacing,pydata = read_file(f_in)
seg_map = lungseg_algorithm.run(img)


# TODO: test with input and output files
#lungseg_algorithm.run_write('opencxr/tests/resources/images/', 'result_folder/')

