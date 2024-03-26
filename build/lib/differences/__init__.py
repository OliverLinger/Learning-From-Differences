"""
The :mod:`differences` module implements the differences
algorithms.
"""

from ._regression import LingerRegressor
from ._classification import LingerClassifier

__all__ = [
    "LingerRegressor",
    "LingerClassifier"
]