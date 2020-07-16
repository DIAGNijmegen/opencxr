# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:38:05 2020

@author: keelin
"""
import os
from pathlib import Path
import string


# TODO: move this elsewhere??
def is_path_safe(path):
    safechars = string.ascii_letters + string.digits + "~ -_."
    path = os.path.normpath(path)
    path_list = path.split(os.sep)
    for item in path_list:
        if any((c not in safechars) for c in item):
            return False
    return True


class BaseAlgorithm():
    valid_extensions = ['.mha', '.MHA', '.mhd', '.MHD', '.png', '.PNG']
    
    def __init__(self):
        pass  # stuff to run on instance creation goes here

    def run(self, input_loc, output_loc):
        # check that the output location does not contain illegal characters (linux would allow creation of files/folders with these)
        if not is_path_safe(output_loc):
            raise FileNotFoundError('Output location contains illegal characters {}'.format(output_loc))
        # if the input location is a folder
        if os.path.isdir(input_loc):
            # Check that the output_location is (or can be) a folder
            os.makedirs(output_loc, exist_ok=True)
            if not os.path.isdir(output_loc):
                raise FileNotFoundError('Could not create output folder at {}'.format(output_loc))
            # check that the input location contains at least one valid file
            found_valid_file = False
            
            for e in self.valid_extensions:   
                if any(fname.endswith(e) for fname in os.listdir(input_loc)):
                    found_valid_file = True
                    break
            if not found_valid_file:
                raise FileNotFoundError('Could not find file with valid extension in {}'.format(input_loc))
            print('folder input and output found valid')
        # if the input location is a file
        elif os.path.isfile(input_loc):
            # Check that the output location is (or can be) a file
            if not os.path.isfile(output_loc):
                Path(output_loc).touch()
            if not os.path.isfile(output_loc):
                raise FileNotFoundError('Could not create file at {}'.format(output_loc))
            print('file input and output found valid')
        else:
            raise FileNotFoundError('Could not find location {}'.format(input_loc))
