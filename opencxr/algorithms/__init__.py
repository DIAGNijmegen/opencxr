# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:31:04 2020

@author: keelin
"""

import importlib
from opencxr.algorithms.base_algorithm import BaseAlgorithm

# names to identify various algorithms available
image_sorter = 'opencxr.algorithms.imagesorter'
lung_seg = 'opencxr.algorithms.lungsegmentation'
heart_seg = 'opencxr.algorithms.heartsegmentation'
cxr_standardize = 'opencxr.algorithms.cxrstandardization'


def create_algorithm(algorithm_modulename):
    """
    Create an instance of a specified algorithm
    :param algorithm_modulename: the algorithm name (from specified list in this file)
    :return: an instance of the algorithm required
    """
    # import the specified algorithm module
    algorithmlib = importlib.import_module(algorithm_modulename)

    # In the file, the class called algorithmfilenamealgorithm() will
    # be instantiated. It has to be a subclass of BaseAlgorithm,
    # and it is case-insensitive.
    algorithm = None
    # take the last part of the algorithm name, e.g. 'lungsegmentation'
    algorithm_name = algorithm_modulename.split('.')[2]
    # add the term 'algorithm' to the end of it
    target_algorithm_name = algorithm_name + 'algorithm'
    # look for the appropriate algorithm class in the imported module
    for name, cls in algorithmlib.__dict__.items():
        if name.lower() == target_algorithm_name.lower() \
                and issubclass(cls, BaseAlgorithm):
            algorithm = cls

    if algorithm is None:
        print("In %s.py, there should be a subclass of BaseAlgorithm with class name that matches %s in lowercase." % (
        algorithm_filename, target_algorithm_name))
        exit(0)

    # get an instance of the identified algorithm
    instance = algorithm()

    print("algorithm [%s] was created" % (instance.name()))
    return instance
