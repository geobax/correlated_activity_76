from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(['project_cython_v8.pyx']),
    include_dirs=[numpy.get_include()]
)