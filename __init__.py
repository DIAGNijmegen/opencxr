# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:18:06 2020

@author: keelin
"""
#import importlib
from opencxr.algorithms import create_algorithm


def load(opencxr_algorithm_name):
    #algorithm = importlib.import_module(opencxr_algorithm_name, package=None)
    algorithm = create_algorithm(opencxr_algorithm_name)
    return algorithm
    
    
#class algorithms:
#    image_sorter = 'opencxr.algorithms.imagesorter'
#    endotr_tube = 'opencxr.algorithms.endotrachealtube'

