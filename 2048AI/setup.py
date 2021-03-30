from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(["game_ai.pyx","game_functions.pyx","game_display.pyx"]),
    include_dirs=[numpy.get_include()]
)   