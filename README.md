## OpenCXR

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

* #### Lung Segmentation:
  Given a PA chest X-ray this algorithm will provide a segmentation of the lung fields as a binary image.  
  This algorithm may work on AP images but is not trained or tested on them.

* #### Heart Segmentation:
  Given a PA chest X-ray this algorithm will provide a segmentation of the heart as a binary image
  This algorithm may work on AP images but is not trained or tested on them.

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
Install conda.  We use [anaconda](https://docs.anaconda.com/anaconda/install/) but miniconda should work too

### Installing
 * clone this repository to your computer
 * get the model weights from XXXXXXXXXXXXXXXXXXX and store them at XXXXXXXXXXXXXXXXXX <span style="font-size:0.9em;">(note that the CXR standardization algorithm does not use a trained neural network and so does not have a model weights file)
 * add the path to the cloned repository to your Python Path
 * use the file opencxr_env.yml (in root folder) to set up a conda environment with the correct packages installed  
   `conda env create --name my_opencxr_env --file opencxr_env.yml`
 * activate the conda environment before running any further commands  
   `conda activate my_opencxr_env`

### Running an algorithm
The easiest way to see how to run the algorithm you are interested in is to look for the algorithm test code in the *tests* folder.
i.e.  
 * tests/test_cxrstandardization.py
 * tests/test_heartsegmentation.py
 * tests/test_imagesorter.py
 * tests/test_lungsegmentation.py

Each of these files contains a minimal code snippet to run the algorithm in question on a sample image.  The principle in each
case is the same: Load the algorithm, read an image, run the algorithm.  The expected returned objects are different depending what algorithm you run.


Note that the heart and lung segmentation algorithms are designed to work on raw PA CXR images, 
however if performance is poor it is likely to be improved by applying CXR standardization to your images first.  

A sample code snippet for lung segmentation is provided below:
```python
import opencxr
from opencxr.utils.file_io import read_file, write_file

# Load the algorithm
lungseg_algorithm = opencxr.load(opencxr.algorithms.lung_seg)
# read an image from disk
img_np, spacing, dicom_tags = read_file('input_path/input_file.mha') # supports mha, mhd, png, dcm
# run the lung segmentation algorithm on the image
seg_map = lungseg_algorithm.run(img_np)
# write the output segmentation to disk
write_file('output_path/output_file.mha', seg_map, spacing)
```

### License ???

OpenCXR is licensed with
the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) @TODO is the license correct? @TODO create a LICENSE file in the repository.


### Questions ???
Do we want to give an email, or just let people create issues?



