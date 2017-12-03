from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy



setup(
	ext_modules=cythonize("get_gt_masks.pyx"),
	include_dirs=[numpy.get_include()]
)