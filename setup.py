#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="shtmbss2",
    version="0.1",
    description="Package containing code for simulating the S-HTM using PyNN with Nest or on BrainScaleS-2",
    url="https://github.com/dietriro/htm-on-bss2",
    author="Robin Dietrich",
    packages=find_packages(exclude=[]),
    install_requires=[
        "colorlog",
        "jupyter",
        "matplotlib",
        "numpy",
        "PyYAML",
        "setuptools",
        "pandas",
        "quantities",
        "PyNN",
        "neo",
        "DateTime",
        "tabulate",
        "nestml==6.0.0",
        "pygsl"
    ],
)
