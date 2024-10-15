# half_edge_utils_setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "half_edge_utils",
        ["half_edge_utils.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="half_edge_utils",
    version="0.1",
    ext_modules=ext_modules,
)
