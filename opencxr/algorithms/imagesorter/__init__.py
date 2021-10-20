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
import wget
import os

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

        # if the file does not exist (it's not included in whl file) then download it from github
        if not os.path.isfile(path_to_model_resolved):
            print('First use of imagesorter model, downloading the weights......')
            file_url = 'https://github.com/DIAGNijmegen/opencxr/tree/master/opencxr/algorithms/model_weights/image_sorter.hdf5'
            os.makedirs(os.path.dirname(path_to_model_resolved), exist_ok=True)
            wget.download(file_url, path_to_model_resolved)
            if not os.path.isfile(path_to_model_resolved):
                print('Failed to download file from', file_url)
                print('Please check the URL is valid and the following location is writeable', path_to_model_resolved)
                return

        self.model = load_model(path_to_model_resolved)

    def run(self, image):
        """
        Run the image sorter algorithm
        :param image: The input image in x,y axis order (as from utils file_io read_image())
        :return: # a dict something like:
        {'Type': 'PA',
           'Rotation': '0',
           'Inversion': 'No',
           'Lateral_Flip': 'No',
           'Type_Probs_PA_AP_lateral_notCXR': [0.99999976, 2.5101654e-08, 2.4382584e-07, 1.0590604e-08],
           'Rotation_Probs_0_90_180_270': [0.9999999, 2.7740466e-08, 2.2800064e-08, 3.7591672e-08],
           'Inversion_Probs_No_Yes': [0.9999968589511354, 3.1410489e-06],
           'Lateral_Flip_Probs_No_Yes': [0.9999986753330177, 1.324667e-06]}
        The first four keys give classifications for Type, Rotation, Inversion, Lateral_Flip.
        The second four keys provide probabilities of all possible classes for users that might need this

        Possible values for the first four keys are as follows:
        Type:  ['PA', 'AP', 'lateral', 'notCXR']
        Rotation: ['0', '90', '180', '270']
        Inversion: ['No', 'Yes']
        Lateral_Flip: ['No', 'Yes']
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
        type_labels = ['PA', 'AP', 'lateral', 'notCXR']
        rotation_labels = ['0', '90', '180', '270']
        inversion_labels = ['No', 'Yes']
        lateral_flip_labels = ['No', 'Yes']

        # return dict
        return {"Type": type_labels[im_type],
                "Rotation": rotation_labels[im_rot],
                "Inversion": inversion_labels[im_inv],
                "Lateral_Flip": lateral_flip_labels[im_flip],
                "Type_Probs_PA_AP_lateral_notCXR": list(pred[0][0]),
                "Rotation_Probs_0_90_180_270": list(pred[1][0]),
                "Inversion_Probs_No_Yes": [1-pred[2][0][0], pred[2][0][0]],
                "Lateral_Flip_Probs_No_Yes": [1-pred[3][0][0], pred[3][0][0]]}
