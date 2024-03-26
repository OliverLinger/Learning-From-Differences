"""
The :mod:`differences` module implements the differences
algorithms.
"""

from ._regression import LingerImageRegressor
from ._classification import LingerImageClassifier

__all__ = [
    "LingerImageRegressor",
    "LingerImageClassifier"
]