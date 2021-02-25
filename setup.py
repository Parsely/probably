#!/usr/bin/env python
from os.path import join

import numpy
from setuptools import setup, Extension


VERSION = '1.1.1'

setup(
    name="probably",
    version=VERSION,
    ext_modules=[Extension("probably.maintenance",
                           sources=[join("probably", "maintenance.c")],
                           include_dirs=[numpy.get_include()])],
)
