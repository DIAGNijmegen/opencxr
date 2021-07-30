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
from skimage import morphology, transform, measure
import imageio
import os
from pathlib import Path
from opencxr.utils.resize_rescale import rescale_to_min_max
from scipy import ndimage

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

	# no longer needed as we use tidy_final_mask
        # pr_test = pr_test > 0.5        
        # post_test = self.post_process(pr_test, remove_ratio * np.prod((input_image.shape[0], input_image.shape[1])))

        segment_map = np.zeros(post_test.shape)
        segment_map[post_test == True] = 255
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

        """
        
        
        if pad_size == 0:
            seg_map_np = transform.resize(seg_map_np, orig_img_shape, order=0)
        
        else:
            # remove the padding from the axis where it was added
            if pad_axis==0:
                left_pad = int(np.round(float(pad_size)/2))
                right_pad = pad_size - left_pad
                seg_map_np = seg_map_np[left_pad:seg_map_np.shape[0]-right_pad,:]
            elif pad_axis==1:
                top_pad = int(np.round(float(pad_size)/2))
                bottom_pad = pad_size - top_pad
                seg_map_np = seg_map_np[:,top_pad:seg_map_np.shape[1]-bottom_pad]
            else:
                print('ERROR: Got a non-zero pad_size (', pad_size, ') but an invalid pad_axis (', pad_axis, ')')
            # Now the padding is removed we can proceed with the resize:
            seg_map_np = transform.resize(seg_map_np, orig_img_shape, order=0)
        """

        resized_seg_map = rescale_to_min_max(resized_seg_map, np.uint8)
        
        return resized_seg_map

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

    def tidy_final_mask(self, lung_mask):
        """
        A method to fill holes and only keep two largest components before returning final mask
        Args:
            image:

        Returns:
            tidied image
        """
        # make binary
        lung_mask = rescale_to_min_max(lung_mask, np.uint8, 0, 1)

        # fill holes
        lung_mask = ndimage.binary_fill_holes(lung_mask)

        # Only keep 2 largest components
        lung_mask = self.get_largest_components(lung_mask, 2).astype(np.uint8)

        lung_mask = rescale_to_min_max(lung_mask, np.uint8, 0, 255)
        return lung_mask
    
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
        # resize the segmentation_512 to the same size as original input
        seg_original = self.resize_to_original(seg_map, size_changes)
        # tidy up by removing holes and keeping two largest connected components
        seg_original = self.tidy_final_mask(seg_original)
        # transpose to return content the same way it was input
        seg_original = np.transpose(seg_original)

        return seg_original

   
        
        
        
        
        
        
    
