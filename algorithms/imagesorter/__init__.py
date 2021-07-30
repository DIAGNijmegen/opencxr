# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:19:57 2020

@author: keelin
"""

from opencxr.algorithms.base_algorithm import BaseAlgorithm
from opencxr.algorithms.imagesorter.preprocess import preprocess_img
from pathlib import Path
from keras.models import load_model
import numpy as np


class ImageSorterAlgorithm(BaseAlgorithm):
    def name(self):
        return 'ImageSorterAlgorithm'

    def __init__(self):
        path_to_model_file = Path(__file__).parent.parent / "model_weights" / "image_sorter.hdf5"
        path_to_model_resolved = str(path_to_model_file.resolve())
        self.model = load_model(path_to_model_resolved)
        
    def run(self, image):
        # first preprocessing:
        image  = preprocess_img(image)

        image = image/255.
        image = np.stack((image, image, image), axis=2)
        pred = self.model.predict(np.expand_dims(image,0))

        im_type = np.argmax(pred[0])
        im_rot = np.argmax(pred[1])
        im_inv = 1 if pred[2] > 0.5 else 0
        im_flip = 1 if pred[3] > 0.5 else 0

        type_labels = ['PA', 'AP', 'lateral', 'not-CXR']
        rotation_labels = ['0', '90', '180', '270']
        inversion_labels = ['No', 'Yes']
        lateral_flip_labels = ['No', 'Yes']

        return {"Type": type_labels[im_type],
                "Rotation": rotation_labels[im_rot],
                "Inversion": inversion_labels[im_inv],
                "Lateral Flip": lateral_flip_labels[im_flip]}

    
