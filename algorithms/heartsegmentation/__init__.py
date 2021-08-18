# -*- coding: utf-8 -*-
"""
Created on Fri July  2 2021

@author: ecem
"""

from opencxr.algorithms.base_algorithm import BaseAlgorithm
from opencxr.algorithms.heartsegmentation.model import unet
import SimpleITK as sitk
import numpy as np
from opencxr.utils import reverse_size_changes_to_img
from opencxr.utils.resize_rescale import rescale_to_min_max, resize_long_edge_and_pad_to_square
from skimage import morphology, transform, measure
import imageio
import os
from scipy import ndimage
from pathlib import Path


class HeartSegmentationAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model =  unet((512,512,1), k_size=3, optimizer='adam', depth = 6,\
            downsize_filters_factor = 2,  batch_norm = True,\
            activation = 'selu', initializer = 'lecun_normal', upsampling = False,\
            dropout = False, n_convs_per_layer = 2, lr=0.00018521094785555384)
        
        # change the path here.
        path_to_model_file = Path(__file__).parent.parent / "model_weights" / "heart_seg.h5"
        path_to_model_resolved = str(path_to_model_file.resolve())
        self.model.load_weights(path_to_model_resolved)
    
    def preprocess(self, image):
        image = image/(np.max(image)/2)-1
        
        mean_train = -0.00557407
        std_train = 0.51691783
        image -= mean_train
        image /= std_train

        return image
    
    def get_largest_components(self, np_array, nr_components=None):
        labels = measure.label(np_array, background=0)

        unique, counts = np.unique(labels, return_counts=True)

        def get_key(item):
            return item[1]

        counts = sorted(zip(unique, counts), key=get_key, reverse=True)
        largest_labels = [c[0] for c in counts if c[0] != 0][0:nr_components]

        if len(largest_labels) == 2:
            return (labels == largest_labels[0]) | (labels == largest_labels[1])
        else:
            return np_array
    
    def tidy_final_mask(self, heart_mask):
        """
        A method to fill holes and only keep two largest components before returning final mask
        Args:
            image:

        Returns:
            tidied image
        """
        # make binary
        heart_mask = rescale_to_min_max(heart_mask, np.uint8, 0, 1)

        # fill holes
        heart_mask = ndimage.binary_fill_holes(heart_mask)

        # Only keep 2 largest components
        heart_mask = self.get_largest_components(heart_mask, 1).astype(np.uint8)

        heart_mask = rescale_to_min_max(heart_mask, np.uint8, 0, 255)
        
        return heart_mask
    

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

        segment_map = np.zeros(pr_test.shape)
        segment_map[pr_test == True] = 255
        segment_map = segment_map.astype(np.uint8)



        return segment_map
    
    
    def name(self):
        return 'HeartSegmentationAlgorithm'
    
    def resize_to_original(self, seg_map_np, size_changes):
        """
        Resize the segmentation map to original dimension.
        """

        # Just reverse the size changes that were applied to the original image
        resized_seg_map, _ = reverse_size_changes_to_img(seg_map_np, [1,1], size_changes, anti_aliasing=False, interp_order=0)
        resized_seg_map = rescale_to_min_max(resized_seg_map, np.uint8)
        
        return resized_seg_map
    
    def run(self, image):
        # transpose because heart segmentation model requires y, x ordering
        image = np.transpose(image)
        
        orig_img_shape = image.shape

        image = np.squeeze(image)
        if len(image.shape)>2 and (image.shape[-1]>1):
            image = np.mean(image, axis=-1)
                
        resized_img, new_spacing, size_changes = resize_long_edge_and_pad_to_square(image, (1,1), 512)
        resized_img = self.preprocess(resized_img)
        seg_map = self.process_image(resized_img)
        
        # if the seg map has nothing segmented then future rescaling operations will fail so we return immediately in that case
        if np.max(seg_map) == 0:
            return np.zeros(orig_img_shape).astype(np.uint8)
        
        
        # resize the segmentation_512 to the same size as original input
        seg_original = self.resize_to_original(seg_map, size_changes)
        # tidy up by removing holes and keeping two largest connected components
        seg_original = self.tidy_final_mask(seg_original)
        # transpose to return content the same way it was input
        seg_original = np.transpose(seg_original)

        return seg_original

   
        
        
        
        
        
        
    
