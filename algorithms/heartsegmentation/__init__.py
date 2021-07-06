# -*- coding: utf-8 -*-
"""
Created on Fri July  2 2021

@author: ecem
"""

from opencxr.algorithms.base_algorithm import BaseAlgorithm
from opencxr.algorithms.lungsegmentation.model import unet
import SimpleITK as sitk
import numpy as np
from opencxr.utils.resize_rescale import resize_long_edge_and_pad_to_square
from skimage import morphology, transform, measure
import imageio
import os


class HeartSegmentationAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model =  unet((512,512,1), k_size=3, optimizer='adam', depth = 6,\
            downsize_filters_factor = 2,  batch_norm = True,\
            activation = 'selu', initializer = 'lecun_normal', upsampling = False,\
            dropout = False, n_convs_per_layer = 2, lr=0.00018521094785555384)
        
        # change the path here.
        self.model.load_weights('opencxr/algorithms/heartsegmentation/model_weights/heart_seg.h5')
    
    def preprocess(self, image):
        image = image/(np.max(image)/2)-1
        
        mean_train = -0.00557407
        std_train = 0.51691783
        image -= mean_train
        image /= std_train

        return image
    
    def post_process(self, mask):
        '''
        Post-processing step for heart segmentation:
        Keeps only the largest connected components of each label for a segmentation mask.
        '''

        out_img = np.zeros(mask.shape, dtype=np.uint8)

        for struc_id in [1]:

            binary_img = mask == struc_id
            blobs = measure.label(binary_img, connectivity=1)

            props = measure.regionprops(blobs)

            if not props:
                continue

            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label

            out_img[blobs == largest_blob_label] = struc_id

        out_img=out_img ==1

        return out_img 
    
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
        post_test = self.post_process(pr_test)

        segment_map = np.zeros(post_test.shape)
        segment_map[post_test == True] = 255


        return sitk.GetImageFromArray(segment_map)
    
    
    def name(self):
        return 'HeartSegmentationAlgorithm'
    
    def resize_to_original(self, seg_map_np, pad_size, pad_axis, orig_img_shape):
        """
        Resize the segmentation map to original dimension.
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
        
        return seg_map_np
    
    def run_filein_fileout(self, input_file:sitk.Image):
        image = sitk.GetArrayFromImage(input_file)
        orig_img_shape = image.shape

        image = np.squeeze(image)
        if len(image.shape)>2 and (image.shape[-1]>1):
            image = np.mean(image, axis=-1)
                
        resized_img, new_spacing, pad_size, pad_axis = resize_long_edge_and_pad_to_square(image, input_file.GetSpacing(), 512)
        resized_img = self.preprocess(resized_img)
        seg_map = self.process_image(resized_img)
        seg_original = self.resize_to_original(sitk.GetArrayFromImage(seg_map), pad_size, pad_axis, orig_img_shape)
        
        return seg_original

   
        
        
        
        
        
        
    
