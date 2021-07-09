# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:31:04 2020

@author: keelin
"""

import importlib
from opencxr.algorithms.base_algorithm import BaseAlgorithm

# names to link with correct packages
image_sorter = 'opencxr.algorithms.imagesorter'
endotr_tube = 'opencxr.algorithms.endotrachealtube'
lung_seg = 'opencxr.algorithms.lungsegmentation'
heart_seg = 'opencxr.algorithms.heartsegmentation'
cxr_standardize = 'opencxr.algorithms.cxrstandardization'


def create_algorithm(algorithm_modulename): 
    algorithmlib = importlib.import_module(algorithm_modulename)
    
    
    # In the file, the class called algorithmfilenamealgorithm() will
    # be instantiated. It has to be a subclass of BaseAlgorithm,
    # and it is case-insensitive.
    algorithm = None
    algorithm_name = algorithm_modulename.split('.')[2]
    target_algorithm_name = algorithm_name + 'algorithm'
    for name, cls in algorithmlib.__dict__.items():
        if name.lower() == target_algorithm_name.lower() \
           and issubclass(cls, BaseAlgorithm):
            algorithm = cls

    if algorithm is None:
        print("In %s.py, there should be a subclass of BaseAlgorithm with class name that matches %s in lowercase." % (algorithm_filename, target_algorithm_name))
        exit(0)

    instance = algorithm()

    print("algorithm [%s] was created" % (instance.name()))
    return instance
