# -*- coding: utf-8 -*-
"""

@author: ecem
"""

from pathlib import Path

import opencxr
from opencxr.utils.file_io import read_file, write_file

# Load the algorithm
lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)
# read an image from disk
f_in = Path(__file__).parent / "resources" / "images" / "c0005.mha"
f_in = str(f_in.resolve())
img_np, spacing, pydata = read_file(f_in)
# run the lung segmentation algorithm on the image
# this will return the segmentation map image
seg_map = lungseg_algorithm.run(img_np)
# write the output segmentation to disk
f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "c0005_lungseg.mha"
f_out = str(f_out.resolve())
write_file(f_out, seg_map, spacing)
