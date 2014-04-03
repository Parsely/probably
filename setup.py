#!/usr/bin/env python
from ez_setup import use_setuptools
use_setuptools()

import os

from setuptools import setup, find_packages, Extension
from distutils.command import build_ext
import numpy

VERSION = '0.0.1'
DESCRIPTION = "PDS: Simple Probabilistic Data Structure"
LONG_DESCRIPTION = """
"""

CLASSIFIERS = filter(None, map(str.strip,
"""
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Operating System :: OS Independent
Topic :: Utilities
Topic :: Database :: Database Engines/Servers
Topic :: Software Development :: Libraries :: Python Modules
""".splitlines()))

setup(
    name="pds",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    keywords=('data structures', 'bloom filter', 'bloom', 'filter',
              'probabilistic', 'set', 'hyperloglog', 'countmin sketch'),
    author="Parse.ly",
    author_email="martin@parse.ly",
    url="https://github.com/Parsely/python-pds",
    license="MIT License",
    packages=find_packages(exclude=['ez_setup']),
    platforms=['any'],
    zip_safe=False,
    install_requires=['numpy', 'cython', 'bitarray', 'smhasher'],
    ext_modules = [Extension("maintenance", ["pds/maintenance.c"], include_dirs=[numpy.get_include()])],
)
