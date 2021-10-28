## OpenCXR

[![Tests](https://github.com/DIAGNijmegen/opencxr/actions/workflows/tests.yml/badge.svg)](https://github.com/DIAGNijmegen/opencxr/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/DIAGNijmegen/opencxr/branch/master/graph/badge.svg)](https://codecov.io/gh/DIAGNijmegen/opencxr)
[![PyPI version](https://badge.fury.io/py/opencxr.svg)](https://badge.fury.io/py/opencxr)
[![PyPI version](https://img.shields.io/badge/codestyle-black-black)](https://img.shields.io/badge/codestyle-black-black)

OpenCXR is an open source collection of chest x-ray (CXR) algorithms and utilities maintained by the 
Diagnostic Image Analysis Group (DIAG) at Radboud University Medical Center, Nijmegen, the Netherlands.
www.diagnijmegen.nl

### OpenCXR Algorithms

The algorithms currently offered are as follows:
* #### Image Sorter: 
  This algorithm is designed to help users sort out volumes of downloaded CXR data. 
  Given a 2D image, Image Sorter will identify it as one of the following image types:  
    * Frontal PA chest X-ray
    * Frontal AP chest X-ray (distinction between PA and AP is not as good as distinction between other types)
    * Lateral chest X-ray
    * Other (not a chest X-ray)
  
  Image Sorter will additionally provide the following useful information for Chest X-Ray images:
    * Rotation (identifies if the image is rotated 90, 180 or 270 degrees)
    * Inversion (identifies if the X-Ray is stored with bright values representing air-filled regions)
    * Lateral Flip (identifies if a lateral Chest X-Ray is stored with the spine to the left of the image)

  Image Sorter is designed to work on raw data and it is not recommended to perform image normalization/standardization procedures in advance.

* #### Lung Segmentation:
  Given a PA chest X-ray this algorithm will provide a segmentation of the lung fields as a binary image.  
  This algorithm may work on AP images but is not trained or tested on them. Performance may be better on images that have been 
  standardized (see _Chest X-ray standardization_ below )

* #### Heart Segmentation:
  Given a PA chest X-ray this algorithm will provide a segmentation of the heart as a binary image.
  This algorithm may work on AP images but is not trained or tested on them.  Performance may be better on images that have been 
  standardized (see _Chest X-ray standardization_ below )

* #### Chest X-ray standardization:
  Given a frontal chest X-ray this algorithm will 
    * standardize the image intensities using energy bands and a region of interest (obtained by lung segmentation).
  This process is described in the following paper: [Localized Energy-Based Normalization of Medical Images: Application to Chest Radiography, Philipsen et. al., IEEE Transactions on Medical Imaging, 2015](https://ieeexplore.ieee.org/document/7073580)
    * (Optionally) crop the image to the lung bounding box
    * Resize the image to a specified square size (e.g. 1024x1024), preserving aspect ratio and padding the shorter axis. 


### OpenCXR Utilities
The utils library contains functions for common or useful functions when working with chest X-rays.  These include:
 * reading/writing images
 * resizing images
 * rescaling intensities
 * masking/cropping
 * rotating/flipping  

In general if you work with OpenCXR algorithms it is recommended to use the OpenCXR image reader/writer and other 
utilities.


## How to use OpenCXR

### Requirements
 - [pip](https://pypi.org/project/pip/) (if you just want to use the algorithms/utilities)
 - [git lfs](https://git-lfs.github.com/) (only if you want to clone this repository including model weights)

### Installation
You can get the latest version of OpenCXR by simply running the command below
```
pip install opencxr
```

### Getting Started
A useful place to get started is to look at the test code for the algorithm you want to run.  You'll find a python file to test each algorithm in the [tests] folder.  
Any algorithm can be loaded by calling my_algorithm = opencxr.load(algorithm_name) . Algorithm names are listed at the 
top of [this file](https://github.com/DIAGNijmegen/opencxr/blob/master/opencxr/algorithms/__init__.py).  
The loaded algorithm can then be run by calling my_algorithm.run(required_input).

The code snippet below gives a suggestion of how to get started with the cxr standardization algorithm. 

```
import opencxr
from opencxr.utils.file_io import read_file, write_file

# an algorithm is created with a call to opencxr.load() .  
cxr_std_algorithm = opencxr.load(opencxr.algorithms.cxr_standardize)

# provide the path to the file you want to read in
full_cxr_file_path = input_folder_cxr + '/' + input_cxr_file

# read the  file (supports dcm, mha, mhd, png)
img_np, spacing, dcm_tags = read_file(full_cxr_file_path)

# Do standardization of intensities, cropping to lung bounding box, and resizing to 1024 square
std_img, new_spacing, size_changes = cxr_std_algorithm.run(img_np, spacing)

# write the standardized file to disk
output_cxr_loc = output_folder_cxr + '/' + input_cxr_file
write_file(output_cxr_loc, std_img, new_spacing)
```


