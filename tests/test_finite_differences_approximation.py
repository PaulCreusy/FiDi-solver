"""
Test model for the finite differences approximation.
"""

###########
# Imports #
###########

# Python imports #

import sys
import numpy as np

# Set the path for local imports
sys.path.append(".")

# Local imports #

from src.finite_differences_approx import compute_finite_difference, compute_taylor_approximation

#########
# Tests #
#########


def test_compute_taylor_approximation():
    taylor_approximation = compute_taylor_approximation(3, 1)
    assert np.isclose(taylor_approximation,
                      np.array([1, 1, 1 / 2, 1 / 6])).all()


def test_compute_finite_difference():
    finite_differentiation = compute_finite_difference(
        target_order=1, inferior_limit=1, superior_limit=1)
    print(finite_differentiation)
    finite_differentiation = compute_finite_difference(
        target_order=2, inferior_limit=2, superior_limit=0)
    print(finite_differentiation)
    finite_differentiation = compute_finite_difference(
        target_order=1, inferior_limit=1, superior_limit=1, derivative=0)
    print(finite_differentiation)

###########
# Process #
###########


if __name__ == "__main__":
    test_compute_taylor_approximation()
    test_compute_finite_difference()
