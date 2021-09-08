# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:19:57 2020

@author: keelin
"""

from pathlib import Path

import numpy as np
from keras.models import load_model
from opencxr.algorithms.base_algorithm import BaseAlgorithm
from opencxr.algorithms.imagesorter.preprocess import preprocess_img

"""
An algorithm to sort 2D images into the following categories:
 - CXR (PA)
 - CXR (AP)
 - CXR (lateral)
 - Non-CXR
In addition, the following information is provided for each image 
 - Rotation (in degrees, based on expectation of an upright CXR image)
 - Lateral flip (for a lateral CXR image, based on expectation of spine to right of lateral image)
 - Inversion (based on expectation of CXR image with dark intensitities for air filled regions)
"""


class ImageSorterAlgorithm(BaseAlgorithm):

    def name(self):
        return 'ImageSorterAlgorithm'

    def __init__(self):
        """
        load the model
        """
        path_to_model_file = Path(__file__).parent.parent / "model_weights" / "image_sorter.hdf5"
        path_to_model_resolved = str(path_to_model_file.resolve())
        self.model = load_model(path_to_model_resolved)

    def run(self, image):
        """
        Run the image sorter algorithm
        :param image: The input image in x,y axis order (as from utils file_io read_image())
        :return: A dict with keys "Type", "Rotation", "Inversion", "Lateral Flip". Possible values for those keys
        are (respectively): ['PA', 'AP', 'lateral', 'not-CXR'], ['0', '90', '180', '270'], ['No', 'Yes'], ['No', 'Yes']
        """
        # first preprocessing:
        image = preprocess_img(image)

        # now get ready and call the model to do prediction
        image = image / 255.
        image = np.stack((image, image, image), axis=2)
        pred = self.model.predict(np.expand_dims(image, 0))

        # binarize the model outputs
        im_type = np.argmax(pred[0])
        im_rot = np.argmax(pred[1])
        im_inv = 1 if pred[2] > 0.5 else 0
        im_flip = 1 if pred[3] > 0.5 else 0

        # set up return labels
        type_labels = ['PA', 'AP', 'lateral', 'not-CXR']
        rotation_labels = ['0', '90', '180', '270']
        inversion_labels = ['No', 'Yes']
        lateral_flip_labels = ['No', 'Yes']

        # return dict
        return {"Type": type_labels[im_type],
                "Rotation": rotation_labels[im_rot],
                "Inversion": inversion_labels[im_inv],
                "Lateral Flip": lateral_flip_labels[im_flip]}
