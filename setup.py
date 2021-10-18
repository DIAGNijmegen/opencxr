#!/usr/bin/env python

import os
from typing import Dict

from setuptools import find_packages, setup

NAME = "opencxr"
REQUIRES_PYTHON = ">=3.7"

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


requirements = [
    #"pandas!=0.24.0",
    #"imageio",
    #"SimpleITK",
    #"cookiecutter",
    #"click",
    #"scipy",
    #"scikit-learn",
    #"numpy",
]

# test_requirements = ["pytest", "pytest-cov", "pytest-xdist", "pytest-randomly"]

# here = os.path.abspath(os.path.dirname(__file__))

setup(
    author="Keelin Murphy",
    author_email="keelin.murphy@radboudumc.nl",
    description=(
        "a collection of algorithms for processing of chest radiograph (CXR) images"
    ),
    install_requires=requirements,
    license="Apache 2.0",
    long_description=readme,
    keywords="opencxr",
    url="https://github.com/DIAGNijmegen/opencxr",
    packages = ['opencxr'],
    version='1.0.0',
)
