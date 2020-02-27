# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:19:57 2020

@author: keelin
"""

from opencxr.algorithms.base_algorithm import BaseAlgorithm


class ImageSorterAlgorithm(BaseAlgorithm):
    def name(self):
        return 'ImageSorterAlgorithm'
        
    # public method, locs can be both filenames or both foldernames
    def run(self, input_loc, output_loc):
        # Do stuff common to all algorithms 
        BaseAlgorithm.run(self, input_loc, output_loc)
        
    # private method to run on individual files
    def __run_filein_fileout(self, input_file_name, output_file_name):
        pass
    
