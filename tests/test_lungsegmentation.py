# -*- coding: utf-8 -*-
"""

@author: ecem
"""

from pathlib import Path
import numpy as np
import opencxr
from opencxr.utils.file_io import read_file, write_file

def run_lung_seg():
    # Load the algorithm
    lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)
    # read an image from disk
    f_in = Path(__file__).parent / "resources" / "images" / "c0005.mha"
    f_in = str(f_in.resolve())
    img_np, spacing, pydata = read_file(f_in)
    # run the lung segmentation algorithm on the image (note: in the wild it may be best to perform cxr standardization first)
    # this will return the segmentation map image
    seg_map = lungseg_algorithm.run(img_np)
    # write the output segmentation to disk
    f_out = Path(__file__).parent / "resources" / "tmp_test_outputs" / "c0005_lungseg.mha"
    f_out = str(f_out.resolve())
    write_file(f_out, seg_map, spacing)

    if seg_map.shape == img_np.shape and not np.max(seg_map) == 0:
        print('Lung Segmentation test completed successfully')
        return 1
    else:
        print('Lung Segmentation results not as expected')
        return 0

def test_lung_seg():
    assert(run_lung_seg()==1)

if __name__=='__main__':
    run_lung_seg()

