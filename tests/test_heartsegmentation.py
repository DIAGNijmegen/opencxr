# -*- coding: utf-8 -*-
"""

@author: ecem
"""

from pathlib import Path
import numpy as np

import opencxr
from opencxr.utils.file_io import read_file, write_file

def run_heart_seg():
    # Load the heart segmentation algorithm
    heartseg_algorithm = opencxr.load(opencxr.algorithms.heart_seg)
    # read in a test image
    f_in = Path(__file__).parent / 'resources' / 'images' / 'c0003.mha'
    f_in = str(f_in.resolve())
    img_np, spacing, pydata = read_file(f_in)
    # run the heart segmentation algorithm (note - in the wild it may be best to perform cxr standardization first)
    # this will return the segmentation map image
    seg_map = heartseg_algorithm.run(img_np)
    # write the output from the segmentation algorithm to disk
    f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "c0003_heartseg.mha"
    f_out = str(f_out.resolve())
    write_file(f_out, seg_map, spacing)

    if seg_map.shape == img_np.shape and not np.max(seg_map) == 0:
        print('Heart Segmentation test completed successfully')
        return 1
    else:
        print('Heart Segmentation results not as expected')
        return 0

def test_heart_seg():
    assert(run_heart_seg() == 1)

if __name__=='__main__':
    run_heart_seg()
