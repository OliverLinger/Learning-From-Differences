from setuptools import setup, find_packages

setup(
    name='differences_images_conv',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.3',
        'pandas>=1.3.3',
        'scikit-learn>=0.24.2',
        'scipy>=1.7.3',
        'tensorflow>=2.6.0',
    ],
)
