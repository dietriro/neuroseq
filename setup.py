#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="neuroseq",
    version="1.0",
    description="Package containing code for simulating ",
    url="https://github.com/dietriro/neuroseq",
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
