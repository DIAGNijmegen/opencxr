# -*- coding: utf-8 -*-
"""

@author: ecem
"""

import opencxr
from pathlib import Path
from opencxr.utils.file_io import read_file, write_file

# Load the algorithm
# possible algorithms are listed in opencxr/algorithms/__init__.py
lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)
# test with numpy array as input
f_in = Path(__file__).parent / "resources" / "images" / "g0536.mha"
f_in = str(f_in.resolve())
img,spacing,pydata = read_file(f_in)
print('img i input to lung seg has shape of ', img.shape)
seg_map = lungseg_algorithm.run(img)
print('seg map i received has shape of', seg_map.shape)
f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "g0536.mha"
f_out = str(f_out.resolve())
print(seg_map.dtype)
write_file(f_out, seg_map, spacing)

