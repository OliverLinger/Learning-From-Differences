"""
The :mod:`differences` module implements the differences
algorithms.
"""

from ._regression import LingerRegressorNewDist
from ._classification import LingerClassifierNewDist

__all__ = [
    "LingerRegressorNewDist",
    "LingerClassifierNewDist"
]