# half_edge_cutils_setup.py
from setuptools import setup, Extension
import pybind11
import os

project_dir = os.path.abspath("./")

ext_modules = [
    Extension(
        "half_edge_cutils",
        [os.path.join(project_dir, "src/cpp/half_edge_cutils.cpp")],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
            os.path.join(project_dir, "include"),
        ],
        language="c++",
    ),
]

setup(
    name="half_edge_cutils",
    version="0.1",
    ext_modules=ext_modules,
    options={"build_ext": {"build_lib": "./src/python/utilities"}},
)
