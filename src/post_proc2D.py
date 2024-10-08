"""
Module to post-process 2D results.
"""

###########
# Imports #
###########

import numpy as np
import matplotlib.pyplot as plt

#############
# Functions #
#############


def plot_contour(X: np.ndarray, Y: np.ndarray, field: np.ndarray, levels: int = 30):
    """
    Plot contour isolines of the given field.
    """

    plt.contour(X, Y, field, levels=levels)
    plt.show()


def plot_surface(X: np.ndarray, Y: np.ndarray, field: np.ndarray):
    """
    Plot the 3D surface of the given field.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, field, cmap=plt.cm.viridis)
    fig.colorbar(surf)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
