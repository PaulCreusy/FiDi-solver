"""
Module to define mesh class and functions.

ToDo
----
- Add possibility to define holes i.e. nodes surrounded by boundary conditions that will not be computed
"""

###########
# Imports #
###########

import numpy as np

###########
# Classes #
###########


class StructuredMesh2D:
    """
    Class to define a 2D structured mesh.

    Parameters
    ----------
    n_x : int
        Number of points of the grid in x direction.
    n_y : int
        Number of points of the grid in y direction;
    l_x : float
        Length of the x side of the grid.
    l_y : float
        Length of the y side of the grid.
    offset_x : float
        Offset for the x coordinates of the grid.
    offset_y : float
        Offset for the y coordinate of the grid.
    """

    def __init__(
            self,
            n_x: int,
            n_y: int,
            l_x: float,
            l_y: float,
            offset_x: float = 0.,
            offset_y: float = 0.) -> None:

        # Define the number of points
        self.n_x = n_x
        self.n_y = n_y

        # Define the length
        self.l_x = l_x
        self.l_y = l_y

        # Define the offset
        self.offset_x = offset_x
        self.offset_y = offset_y

    def update_grid(self):
        """
        Update the grid of the mesh.
        """
        self.x = np.linspace(0, self.l_x, self.n_x) + self.offset_x
        self.y = np.linspace(0, self.l_y, self.n_y) + self.offset_y
        self.grid_x, self.grid_y = np.meshgrid(self.x, self.y, indexing='ij')

    def __getattribute__(self, i, j) -> np.Any:
        return np.concatenate([self.grid_x[i, j], self.grid_y[i, j]])

    @property
    def shape(self):
        return self.grid_x.shape
