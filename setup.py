#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """
    This class is necessary because numpy won't be installed at import time.
    """
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


VERSION = '1.0.0'
DESCRIPTION = "Probably: Simple Probabilistic Data Structures"
LONG_DESCRIPTION = ""
CLASSIFIERS = ['Intended Audience :: Developers',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python',
               'Operating System :: OS Independent',
               'Topic :: Utilities',
               'Topic :: Database :: Database Engines/Servers',
               'Topic :: Software Development :: Libraries :: Python Modules']

setup(
    name="probably",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    keywords=['data structures', 'bloom filter', 'bloom', 'filter',
              'probabilistic', 'set', 'hyperloglog', 'countmin sketch'],
    author="Parse.ly",
    author_email='hello@parsely.com',
    url="https://github.com/Parsely/probably",
    license="MIT License",
    packages=find_packages(),
    platforms=['any'],
    zip_safe=False,
    install_requires=['numpy', 'cython', 'bitarray', 'six', 'smhasher'],
    setup_requires=['numpy'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("probably.maintenance", ["probably/maintenance.c"])],
)
