from pathlib import Path
import os
import runpy
import sys
import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

root = Path(__file__).resolve().parent
os.chdir(root)
if '--numba' in sys.argv:
    runpy.run_path(str(root/'mBEST/utils/numba_utils.py'), run_name='__main__')
else:
    sys.argv = [sys.argv[0], 'build_ext', '--inplace']
    ext = Extension('mBEST.skeletonize.skeletonize', ['mBEST/skeletonize/skeletonize.pyx'], include_dirs=[numpy.get_include()])
    setup(ext_modules=cythonize([ext], language_level=3))
