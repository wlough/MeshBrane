# cply_tools_setup.py
from setuptools import setup, Extension
import pybind11
import os

# python cply_tools_setup.py build_ext --inplace

project_dir = os.path.abspath("./")

ext_modules = [
    Extension(
        "cply_tools",
        [os.path.join(project_dir, "src/cpp/cply_tools.cpp")],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
            os.path.join(project_dir, "include"),
        ],
        language="c++",
    ),
]

setup(
    name="cply_tools",
    version="0.1",
    ext_modules=ext_modules,
    options={"build_ext": {"build_lib": "./src/python/mesh"}},
)
