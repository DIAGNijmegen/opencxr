#!/usr/bin/env python

import os
from typing import Dict

from setuptools import find_packages, setup

NAME = "opencxr"
REQUIRES_PYTHON = ">=3.7"

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


requirements = [
    # "pandas!=0.24.0",
    # "imageio",
    # "cookiecutter",
    # "click",
    # "scipy",
    "wget",
    "tensorflow",
    "SimpleITK",
    "keras",
    "pydicom",
    "pypng",
    "scikit-image",
    "scikit-build",
    "numpy",
]

# test_requirements = ["pytest", "pytest-cov", "pytest-xdist", "pytest-randomly"]

# here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="opencxr",
    author="Keelin Murphy",
    author_email="keelin.murphy@radboudumc.nl",
    description=(
        "a collection of algorithms for processing of chest radiograph (CXR) images"
    ),
    install_requires=requirements,
    # include_package_data=True,
    license="Apache 2.0",
    long_description="a collection of algorithms for processing of chest radiograph (CXR) images",
    keywords="opencxr",
    url="https://github.com/DIAGNijmegen/opencxr",
    packages=find_packages(),
    version="1.2.0",
)
