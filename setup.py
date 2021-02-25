#!/usr/bin/env python
from os.path import join

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

VERSION = "1.1.1"

extensions = [
    Extension(
        "probably.maintenance",
        [join("probably", "maintenance.pyx")],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="probably",
    version=VERSION,
    setup_requires=["oldest-supported-numpy", "cython"],
    ext_modules=cythonize(extensions),
)
