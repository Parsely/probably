#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


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
    platforms=['any'],
    zip_safe=False,
    cmdclass={'build_ext':build_ext},
    setup_requires=['numpy'],
    install_requires=['numpy', 'cython', 'bitarray', 'smhasher'],
    ext_modules = [Extension("maintenance", ["pds/maintenance.c"]),
                   Extension("bloomfilter_cython", ["pds/bloomfilter_cython.c", "pds/MurmurHash2A.c", "pds/MurmurHash3.cpp"]),

    ],
)
