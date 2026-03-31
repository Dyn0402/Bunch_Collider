"""
Build script for the pybind11 C++ extension.

Usage
-----
Install in editable mode (recommended for development)::

    pip install -e .

Or build the extension in-place (useful for testing without installing)::

    python setup.py build_ext --inplace
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bunch_collider._bunch_density_cpp",
        ["bunch_collider/_bunch_density_cpp.cpp"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
