# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:18:06 2020

@author: keelin
"""

from opencxr.algorithms import create_algorithm


def load(opencxr_algorithm_name):
    algorithm = create_algorithm(opencxr_algorithm_name)
    return algorithm


