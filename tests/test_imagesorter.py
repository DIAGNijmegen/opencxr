# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:40:35 2020

@author: keelin
"""

import opencxr
from pathlib import Path

# Load the algorithm
# possible algorithms are listed in opencxr/algorithms/__init__.py
lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)

# test with input and output folders
f_in = '/mnt/synology/cxr/projects/cxr-cardiomegaly_t7327/data/evaluation/evaluation_images/'
f_out = '/mnt/synology/cxr/temp/ecem/results_opencxr_lung/'
lungseg_algorithm.run(f_in, f_out)

# TODO: test with input and output files
