"""
The :mod:`differences` module implements the differences
algorithm.
"""

from ._regression import LingerRegressor
from ._classification import LingerClassifier

__all__ = [
    "LingerRegressor",
    "LingerClassifier"
]