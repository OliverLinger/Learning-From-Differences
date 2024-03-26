from setuptools import setup, find_packages

setup(
    name='differences_models',  # Change this to the name of your project
    version='0.1',  # Update with your project version
    packages=find_packages(),  # Automatically find packages in the project directory
    install_requires=[  # List dependencies required for your project
        'numpy',
        'scikit-learn',
        # Add other dependencies as needed
    ],
    # Additional metadata
    author='Your Name',
    author_email='your@email.com',
    description='Description of your project',
    url='https://github.com/OliverLinger/fyp',  # Update with your project URL
    classifiers=[  # Classifiers provide information about your project
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)