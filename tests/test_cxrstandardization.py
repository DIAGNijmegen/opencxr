# -*- coding: utf-8 -*-
"""

@author: keelin
"""

import opencxr
from pathlib import Path
from opencxr.utils.file_io import read_file, write_file
from opencxr.utils.mask_crop import crop_img_borders
from opencxr.utils import apply_size_changes_to_img
from PIL import Image
import os
import numpy as np


# read in and standardize an image
f_in = Path(__file__).parent / "resources" / "images" / "g0019.mha"
f_in = str(f_in.resolve())
img_np,spacing,pydata = read_file(f_in)
cxrstandardize_algorithm = opencxr.load(opencxr.algorithms.cxr_standardize)
cxrstandardize_algorithm
final_norm_img, new_spacing, size_changes = cxrstandardize_algorithm.run(img_np, spacing)
print('size changes was', size_changes)
f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "g0019_norm.mha"
f_out = str(f_out.resolve())
write_file(f_out, final_norm_img, new_spacing)

# now make a lung segmentation for the original image and try to resize it using the size_changes information
lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)
seg_map_fullsize = lungseg_algorithm.run(img_np)
seg_map_resized_for_norm, new_lung_seg_spacing = apply_size_changes_to_img(seg_map_fullsize, spacing, size_changes, anti_aliasing=False, interp_order=0)

f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "g0019_lungseg.mha"
f_out = str(f_out.resolve())
write_file(f_out, seg_map_fullsize, spacing)
f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "g0019_norm_lungseg.mha"
f_out = str(f_out.resolve())
write_file(f_out, seg_map_resized_for_norm, new_lung_seg_spacing)

#just for interest also lung seg the final norm image to see how it looks by comparison to the resized original
seg_map_normsize = lungseg_algorithm.run(final_norm_img)
f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "g0019_norm_lungsegonnorm.mha"
f_out = str(f_out.resolve())
write_file(f_out, seg_map_normsize, new_lung_seg_spacing)

