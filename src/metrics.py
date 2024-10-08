"""
Module defining functions to compute metrics.
"""

###########
# Imports #
###########

import numpy as np
from numba import jit

#############
# Functions #
#############


@jit(nopython=True)
def compute_L2_error(num_vals: np.ndarray, ana_vals: np.ndarray, normalize=True):
    """
    Compute the L2 error between the given numerical and analytical values.
    """

    # Compute the squared error matrix
    squared_diff = np.square(num_vals - ana_vals)

    # Normalize if needed
    if normalize:
        squared_diff = squared_diff / np.sum(np.square(ana_vals))

    # Average the error over the matrix
    error = np.sum(squared_diff) / num_vals.size

    return error
