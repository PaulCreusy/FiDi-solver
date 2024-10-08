"""
Model to define a partial differential equation problem formulation.

ToDO
----
- Generalize the approach for non-uniform meshes.
"""

###########
# Imports #
###########

# Python imports #

from typing import Callable, Literal
import matplotlib.pyplot as plt
import numpy as np

# Local imports #

from .mesh import StructuredMesh2D

###########
# Classes #
###########


class BoundaryCondition:
    def __init__(
            self,
            name: str,
            bc_type: Literal["dirichlet", "neumann"],
            domain_function: Callable,
            value_function: Callable) -> None:
        self.name = name
        self.bc_type = bc_type
        self.domain_function = domain_function
        self.value_function = value_function


class Problem2D:
    """
    Class to define a 2D differential equations problem.
    """

    def __init__(self, mesh: StructuredMesh2D) -> None:
        self.mesh = mesh
        self.boundary_conditions = {}

    def define_boundary_condition(self, name: str, bc_type: Literal["dirichlet", "neumann"], domain_function: Callable, value_function: Callable):
        self.boundary_conditions[name] = BoundaryCondition(
            name=name,
            bc_type=bc_type,
            domain_function=domain_function,
            value_function=value_function
        )

    def plot_boundary_condition(self, name: str, show_plot=True):
        boundary_points = np.zeros(self.mesh.shape, dtype=np.bool8)
        for i in range(self.mesh.n_x):
            for j in range(self.mesh.n_y):
                boundary_points[i, j] = 1
        boundary_points_list = np.where(boundary_points)
        plt.scatter(self.mesh.grid_x[boundary_points_list],
                    self.mesh.grid_y[boundary_points_list], label=name)

        # Show the plot if needed
        if show_plot:
            plt.legend()
            plt.show()

    def plot_all_boundary_conditions(self):
        for boundary_condition_name in self.boundary_conditions:
            boundary_points = np.zeros(self.mesh.shape, dtype=np.bool8)
            for i in range(self.mesh.n_x):
                for j in range(self.mesh.n_y):
                    boundary_points[i, j] = 1
            boundary_points_list = np.where(boundary_points)
            plt.scatter(self.mesh.grid_x[boundary_points_list],
                        self.mesh.grid_y[boundary_points_list], label=boundary_condition_name)
        plt.legend()
        plt.show()
