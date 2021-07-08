# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:38:05 2020

@author: keelin
"""


class BaseAlgorithm():
    
    def __init__(self):
        pass  # stuff to run on instance creation goes here
    
    def run(self, input_loc):
        return self.run_data_in_data_out(input_loc)
