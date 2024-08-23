from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("half_edge_base_cython_utils.py"),
)
