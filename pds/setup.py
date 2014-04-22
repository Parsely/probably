from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("bloomfilter_cython", ["bloomfilter_cython.pyx", "MurmurHash2A.c", "MurmurHash3.cpp"]),
                   Extension("maintenance", ["maintenance.pyx"], include_dirs=[numpy.get_include()]),
    ]
)


