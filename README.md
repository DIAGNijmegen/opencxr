## OpenCXR 
OpenCXR is an open source collection of chest x-ray (CXR) algorithms. OpenCXR comes with two packages, OpenCXR-Core and OpenCXR-Utils. 

### OpenCXR-Core
The core library lets users run algorithms in an easy way.

### OpenCXR-Utils
The utils library lets users create algorithms in an easy way. It includes python scripts for various operations that are necessary for image manipulation.

## How to use OpenCXR
### Requirements
To be able to use OpenCXR, [install docker](https://runnable.com/docker/getting-started/).

Install OpenCXR using pip `pip install opencxr`.

To access an OpenCXR algorithm in python, use 

### Usage
OpenCXR is licensed with the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) and make sure your purposes are not conflicting with it. 
@TODO is the license correct?
@TODO create a LICENSE file in the repository.

To use OpenCXR, simply import it in python, load the necessary algorithm (see [algorithms](algorithms.md))

```python
import opencxr
import numpy as np

# Load the algorithm
lung_segmentation_algorithm = opencxr.load(opencxr.algorithms.lung_segmentation)

# Use the algorithm directly with data
mock_data = np.zeros((224,224))
segmentation_result = lung_segmentation_algorithm.segment(mock_data)

# Use the algorithm on a set of images 
segmentation_results = lung_segmentation_algorithm.segment('/path/to/folder')
```

See the corresponding algorithm documentation for supported functionality.


## Contributing
You can contribute to OpenCXR in 4 ways.

### 1. Contributing an Algorithm to OpenCXR

#### Requirements
In grand-challenge.org, OpenCXR hosts some challenges. To see the list of hosted challenges, go to the [grand-challenge challenges page](https://grand-challenge.org/challenges/) and choose OpenCXR from the host list. Contribution to the OpenCXR is open for the contributors of these challenges. There are a few requirements to be able to contribute your algorithm to OpenCXR.
1. @TODO Details about the license
1. The algorithm code has to be published in github.com
1. @TODO Decisions about reproducability, do we require being able to retrain the models(UOKs)? 

#### How to contribute

1. Publish the results of your algorithm under the relevant challenge.
1. If you haven't already, create a public repository in github.com and push your code in it. For more information, see [Getting Started with Git Basics](https://git-scm.com/book/en/v1/Getting-Started-Git-Basics).
1. Create a Dockerfile that executes your algorithm and produces the prediction file.
1. Create a branch in your github repository. This branch should only contain the Dockerfile and the necessary files to run this script (such as weights). Please try to minimize the amount of code in this branch by removing unnecessary files, and code blocks. 
1. Create a pull request to OpenCXR, mention the algorithm

### 2. Contributing a Challenge to OpenCXR
If you think that OpenCXR should include a clinically relevant algorithm or an algorithm that would ease the  You can contribute a challenge to OpenCXR. 
1. As explained in grand-challenge.org, [create your own challenge](https://grand-challenge.org/Support/).
1. Add OpenCXR as an admin. @TODO We should Ask James about how these two steps should be.
1. Create an issue under the OpenCXR github repository. We will review your request and discuss the situation further.


### 3. Contributing Data to OpenCXR
Please get in touch with opencxr@gmail.com for contributing data to OpenCXR. The process generally involves parties signing a Data Transfer Agreement (DTO). 


### 4. Contributing to the OpenCXR-utils or OpenCXR-core
See the [utils contribution guide](utils/CONTRIBUTING.md) and [core contribution guide](core/CONTRIBUTING.md).