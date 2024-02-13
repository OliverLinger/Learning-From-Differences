# mymlmodel/setup.py
from setuptools import setup, find_packages

setup(
    name='dummymodel',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        # Add other dependencies as needed
    ],
)