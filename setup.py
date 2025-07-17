# setup.py
from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
)