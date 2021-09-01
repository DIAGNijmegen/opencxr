# -*- coding: utf-8 -*-
"""
Created on Fri July  2 2021

@author: ecem
"""

from opencxr.algorithms.base_algorithm import BaseAlgorithm
from opencxr.algorithms.lungsegmentation.model import unet
import numpy as np
from opencxr.utils import reverse_size_changes_to_img
from opencxr.utils.resize_rescale import resize_long_edge_and_pad_to_square
from skimage import morphology, transform
import imageio
import os
from pathlib import Path
from opencxr.utils.resize_rescale import rescale_to_min_max
from opencxr.utils.mask_crop import tidy_segmentation_mask


class LungSegmentationAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model =  unet((512,512,1), k_size=3, optimizer='RMSprop', depth = 6,\
            downsize_filters_factor = 2,  batch_norm = True,\
            activation = 'elu', initializer = 'he_normal', upsampling = True,\
            dropout = False, n_convs_per_layer = 1, lr=0.0007658078359138082)
        
        path_to_model_file = Path(__file__).parent.parent / "model_weights" / "lung_seg.h5"
        path_to_model_resolved = str(path_to_model_file.resolve())
        self.model.load_weights(path_to_model_resolved)
    
    def preprocess(self, image):
        image = image/(np.max(image)/2)-1
        
        mean_train = -0.00557407
        std_train = 0.51691783
        image -= mean_train
        image /= std_train

        return image
    
    # no longer needed as we use tidy_final_mask
    """
    def post_process(self, img, size):
        # Morphologically removes small (less than size) connected regions of 0s or 1s.

        img = morphology.remove_small_objects(img, size)
        img = morphology.remove_small_holes(img, size)
        return img
    """
    
    def process_image(self, input_image, remove_ratio = 0.05):
        '''
        Produce segmentation map prediction of the network.

        1. Get the network prediction.
        2. Apply post-processing choosing the largest connected component.

        '''
        input_image = input_image.astype(np.float32)
        input_image = np.expand_dims(input_image, -1)

        if len(input_image.shape)==3:
            input_image = np.expand_dims(input_image, 0)


        pr_test = self.model.predict(input_image).squeeze()


        pr_test = pr_test > 0.5

        # no longer needed as we use tidy_final_mask
        # post_test = self.post_process(pr_test, remove_ratio * np.prod((input_image.shape[0], input_image.shape[1])))

        segment_map = np.zeros(pr_test.shape)
        segment_map[pr_test == True] = 255
        segment_map = segment_map.astype(np.uint8)


        return segment_map
    
    
    def name(self):
        return 'LungSegmentationAlgorithm'
    
    def resize_to_original(self, seg_map_np, size_changes):
        """
        Resize the segmentation map to original dimension.
        """

        # Just reverse the size changes that were applied to the original image
        resized_seg_map, _ = reverse_size_changes_to_img(seg_map_np, [1,1], size_changes, anti_aliasing=False, interp_order=0)

        resized_seg_map = rescale_to_min_max(resized_seg_map, np.uint8)
        
        return resized_seg_map

    
    def run(self, image):
        """

        Args:
            image: np array, expected with x, y ordering

        Returns:
            lung_seg_image: np array, x,y ordering, with 0 for background and 255 for lung

        """
        # transpose because lung segmentation model requires y, x ordering
        image = np.transpose(image)

        orig_img_shape = image.shape

        image = np.squeeze(image)
        if len(image.shape)>2 and (image.shape[-1]>1):
            image = np.mean(image, axis=-1)

        # resize to 512 to fit the model (don't care about spacing here)
        resized_img, new_spacing, size_changes = resize_long_edge_and_pad_to_square(image, (1,1), 512)
        # do some preprocessing (intensity normalization)
        resized_img = self.preprocess(resized_img)
        # get the segmentation_512
        seg_map = self.process_image(resized_img)

        # if the seg map has nothing segmented then future rescaling operations will fail so we return immediately in that case
        if np.max(seg_map) == 0:
            return np.zeros(orig_img_shape).astype(np.uint8)

        # resize the segmentation_512 to the same size as original input
        seg_original = self.resize_to_original(seg_map, size_changes)
        # tidy up by removing holes and keeping two largest connected components
        seg_original = tidy_segmentation_mask(seg_original, nr_components_to_keep=2)
        # transpose to return content the same way it was input
        seg_original = np.transpose(seg_original)

        return seg_original

   
        
        
        
        
        
        
    
