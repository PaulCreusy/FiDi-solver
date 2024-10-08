"""
Module to define a simple iterative solver structure.
"""

###########
# Imports #
###########

# Python imports #

import numpy as np

# Local imports #

from .problem import Problem2D
from .numerical_scheme import NumericalScheme2D

###########
# Classes #
###########


class Solver2D:
    def __init__(self, problem: Problem2D, numerical_scheme: NumericalScheme2D, target_precision: float = 1e-6) -> None:
        self.mesh = problem.mesh
        self.numerical_scheme = numerical_scheme
        self.previous_sol = np.zeros(self.mesh.shape, dtype=np.float64)
        self.next_sol = np.zeros(self.mesh.shape, dtype=np.float64)
        self.target_precision = target_precision

        # Create a new structure to access the boundary conditions
        self.boundary_condition_matrix = np.zeros(
            self.mesh.shape, dtype=np.int16)
        self.dirichlet_values = np.zeros(
            self.mesh.shape, dtype=np.float64)
        for boundary_condition in problem.boundary_conditions:
            pass

    def solve():
        pass

    def compute_residual(self):
        residual = 0.
        for i in range(self.mesh.n_x):
            for j in range(self.mesh.n_y):
                pass
        pass


class JacobiSolver2D(Solver2D):
    def solve():
        pass
