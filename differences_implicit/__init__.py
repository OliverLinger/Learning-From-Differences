"""
The :mod:`differences_implicit` module implements the differences
algorithms, as an ensemble of specialised neural netowrks.
"""

from ._regression import LingerImplicitRegressor
from ._classification import LingerImplicitClassifier

__all__ = [
    "LingerImplicitRegressor",
    "LingerImplicitClassifier"
]