# chalf_edge_setup.py
from setuptools import setup, Extension
import pybind11
import os

# python chalf_edge_setup.py build_ext --inplace
name = ("chalf_edge",)
project_dir = os.path.abspath("./")
build_dir = os.path.join(project_dir, "temp_python/build")
lib_dir = os.path.join(project_dir, "temp_python/lib")
for _ in (build_dir, lib_dir):
    os.makedirs(_, exist_ok=True)


chalf_edge_ext_modules = [
    Extension(
        "chalf_edge",
        [
            os.path.join(project_dir, "temp_python/setup/chalf_edge.cpp"),
            os.path.join(project_dir, "temp_python/setup/cply_tools.cpp"),
            os.path.join(project_dir, "temp_python/setup/chalf_edge_pybind.cpp"),
        ],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
            os.path.join(project_dir, "temp_python/include"),
        ],
        extra_compile_args=["-std=c++20", "-fcoroutines"],
        language="c++",
    ),
]

setup(
    name="chalf_edge",
    version="0.1",
    ext_modules=chalf_edge_ext_modules,
    options={
        "build": {"build_base": build_dir},
        "build_ext": {"build_lib": lib_dir},
    },
)
