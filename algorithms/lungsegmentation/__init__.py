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
from skimage import morphology
import imageio
import os

class LungSegmentationAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model =  unet((512,512,1), k_size=3, optimizer='RMSprop', depth = 6,\
            downsize_filters_factor = 2,  batch_norm = True,\
            activation = 'elu', initializer = 'he_normal', upsampling = True,\
            dropout = False, n_convs_per_layer = 1, lr=0.0007658078359138082)
        
        # change the path here.
        self.model.load_weights('/mnt/synology/cxr/projects/cxr-cardiomegaly_t7327/processor/modelweights/hyperopt_lung_best_weights.h5')
    
    def preprocess(self, image):
        image = image/(np.max(image)/2)-1
        
        mean_train = -0.00557407
        std_train = 0.51691783
        image -= mean_train
        image /= std_train

        return image
    
    def post_process(self, img, size):
        """Morphologically removes small (less than size) connected regions of 0s or 1s."""

        img = morphology.remove_small_objects(img, size)
        img = morphology.remove_small_holes(img, size)
        return img
    
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
        post_test = self.post_process(pr_test, remove_ratio * np.prod((input_image.shape[0], input_image.shape[1])))

        segment_map = np.zeros(post_test.shape)
        segment_map[post_test == True] = 255


        return sitk.GetImageFromArray(segment_map)
    
    
    def name(self):
        return 'LungSegmentationAlgorithm'
        
        
    # method to run on individual files
    def run_filein_fileout(self, filename, input_file:sitk.Image, output_file):
        image = sitk.GetArrayFromImage(input_file)
        image = np.squeeze(image)
        if len(image.shape)>2 and (image.shape[-1]>1):
            image = np.mean(image, axis=-1)
                
        resized_img, new_spacing = resize_long_edge_and_pad_to_square(image, input_file.GetSpacing(), 512)
        resized_img = self.preprocess(resized_img)
        seg_map = self.process_image(resized_img)
        imageio.imwrite(os.path.join(output_file, filename+'.png'), sitk.GetArrayFromImage(seg_map))
        
        
        
        
        
        
        
        
    
