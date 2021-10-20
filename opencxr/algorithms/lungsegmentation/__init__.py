# -*- coding: utf-8 -*-
"""
Created on Fri July  2 2021

@author: ecem
"""

from pathlib import Path

import numpy as np
from opencxr.algorithms.base_algorithm import BaseAlgorithm
from opencxr.algorithms.lungsegmentation.model import unet
from opencxr.utils import reverse_size_changes_to_img
from opencxr.utils.mask_crop import tidy_segmentation_mask
from opencxr.utils.resize_rescale import rescale_to_min_max
from opencxr.utils.resize_rescale import resize_long_edge_and_pad_to_square
import wget
import os

"""
Segments the lungs in frontal CXR images
Performance may be better on images that have had cxrstandardization, depending on original image appearance 
"""

class LungSegmentationAlgorithm(BaseAlgorithm):
    def __init__(self):
        """
        initialize the model and weights
        """
        self.model = unet((512, 512, 1), k_size=3, optimizer='RMSprop', depth=6, \
                          downsize_filters_factor=2, batch_norm=True, \
                          activation='elu', initializer='he_normal', upsampling=True, \
                          dropout=False, n_convs_per_layer=1, lr=0.0007658078359138082)

        path_to_model_file = Path(__file__).parent.parent / "model_weights" / "lung_seg.h5"
        path_to_model_resolved = str(path_to_model_file.resolve())

        # if the file does not exist (it's not included in whl file) then download it from github
        if not os.path.isfile(path_to_model_resolved):
            print('First use of lung segmentation model, downloading the weights......')
            file_url = 'https://github.com/DIAGNijmegen/opencxr/tree/master/opencxr/algorithms/model_weights/lung_seg.h5'
            os.makedirs(os.path.dirname(path_to_model_resolved), exist_ok=True)
            wget.download(file_url, path_to_model_resolved)
            if not os.path.isfile(path_to_model_resolved):
                print('Failed to download file from', file_url)
                print('Please check the URL is valid and the following location is writeable', path_to_model_resolved)
                return

        self.model.load_weights(path_to_model_resolved)

    def preprocess(self, image):
        """
        Basic mean and std-dev pre processing
        :param image: np array image
        :return: preprocessed np array image
        """
        image = image / (np.max(image) / 2) - 1

        mean_train = -0.00557407
        std_train = 0.51691783
        image -= mean_train
        image /= std_train

        return image

    def process_image(self, input_image, remove_ratio=0.05):
        """
        Calls the model to obtain lung segmentation
        :param input_image: input np array 512x512 y,x ordering
        :return: np array 512x512 of lung segmentation
        """

        input_image = input_image.astype(np.float32)
        input_image = np.expand_dims(input_image, -1)

        if len(input_image.shape) == 3:
            input_image = np.expand_dims(input_image, 0)

        pr_test = self.model.predict(input_image).squeeze()

        pr_test = pr_test > 0.5

        segment_map = np.zeros(pr_test.shape)
        segment_map[pr_test == True] = 255
        segment_map = segment_map.astype(np.uint8)

        return segment_map

    def name(self):
        return 'LungSegmentationAlgorithm'

    def resize_to_original(self, seg_map_np, size_changes):
        """
        Resize the segmentation map (512x512) to original image size
        :param seg_map_np: the segmentation map
        :param size_changes: the size changes that were applied to the original image (obtained from utils resize calls)
        :return: the segmentation map with size matching original input
        """
        # Just reverse the size changes that were applied to the original image
        resized_seg_map, _ = reverse_size_changes_to_img(seg_map_np, [1, 1], size_changes, anti_aliasing=False,
                                                         interp_order=0)

        resized_seg_map = rescale_to_min_max(resized_seg_map, np.uint8)

        return resized_seg_map

    def run(self, image):
        """
        The call to run lung segmentation on an image
        :param image: The input image as np array, x, y ordering (eg from utils file_io read_file)
        :return: The segmentation image as np array, same size as input image
        """
        # transpose because lung segmentation model requires y, x ordering
        image = np.transpose(image)

        orig_img_shape = image.shape

        image = np.squeeze(image)
        if len(image.shape) > 2 and (image.shape[-1] > 1):
            image = np.mean(image, axis=-1)

        # resize to 512 to fit the model
        resized_img, new_spacing, size_changes = resize_long_edge_and_pad_to_square(image, (1, 1), 512)
        # do some basic preprocessing
        resized_img = self.preprocess(resized_img)
        # get the segmentation in size 512
        seg_map = self.process_image(resized_img)

        # if the seg map has nothing segmented then future rescaling operations will fail so we return immediately in that case
        if np.max(seg_map) == 0:
            return np.zeros(orig_img_shape).astype(np.uint8)

        # resize the segmentation (512x512) to the same size as original input using record of original size changes
        seg_original = self.resize_to_original(seg_map, size_changes)
        # tidy up by removing holes and keeping two largest connected components
        seg_original = tidy_segmentation_mask(seg_original, nr_components_to_keep=2)
        # transpose to return content the same way it was input
        seg_original = np.transpose(seg_original)

        return seg_original
