"""
Model to define a partial differential equation problem formulation.

ToDO
----
Implement Robin boundary conditions.
"""

###########
# Imports #
###########

# Python imports #

from copy import copy
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# Local imports #

from .mesh import StructuredMesh2D
from .finite_differences_approx import compute_finite_difference

###########
# Classes #
###########


class BoundaryCondition:
    def __init__(self, name: str, domain_function: Callable, value_function: Callable) -> None:
        self.name = name
        self.domain_function = domain_function
        self.value_function = value_function


class Term2D:

    x_derivation_order: int = 0
    y_derivation_order: int = 0
    h_power: int = 0
    scalar_factor: int = 1

    def dx(self):
        """
        Return the y derivative of the term.

        Returns
        -------
        Term2D
            Y derivative of the term.
        """

        self_copy = copy(self)
        self_copy.x_derivation_order += 1

        return self_copy

    def dy(self):
        """
        Return the y derivative of the term.

        Returns
        -------
        Term2D
            Y derivative of the term.
        """

        self_copy = copy(self)
        self_copy.y_derivation_order += 1

        return self_copy

    def get_discretized(
            self,
            stencil_inferior_limit: int,
            stencil_superior_limit: int,
            convergence_order: int):
        """
        Compute a discretized version of the term.

        Parameters
        ----------
        stencil_inferior_limit : int
            Inferior limit of the stencil.
        stencil_superior_limit : int
            Superior limit of the stencil.
        convergence_order : int
            Order of convergence.

        Returns
        -------
        DiscretizedTerm2D
            Discretized version of the term.
        """

        # Compute a finite difference approximation for the x axis
        x_finite_diff_approx = compute_finite_difference(
            target_order=convergence_order,
            inferior_limit=stencil_inferior_limit,
            superior_limit=stencil_superior_limit,
            derivative=self.x_derivation_order
        )

        # Compute a finite difference approximation for the y axis
        y_finite_diff_approx = compute_finite_difference(
            target_order=convergence_order,
            inferior_limit=stencil_inferior_limit,
            superior_limit=stencil_superior_limit,
            derivative=self.y_derivation_order
        )

        # Extend dimension
        x_finite_diff_approx = x_finite_diff_approx.reshape(
            (x_finite_diff_approx.size, 1))
        y_finite_diff_approx = y_finite_diff_approx.reshape(
            (1, y_finite_diff_approx.size))

        # Insert the coefficients in the stencil
        stencil = x_finite_diff_approx @ y_finite_diff_approx

        # Create the discretized term
        discretized_term = DiscretizedTerm2D(
            stencil=stencil,
            h_power=-convergence_order
        )

        return discretized_term


class DiscretizedTerm2D:

    def __init__(self, stencil: np.ndarray, h_power: int = 0) -> None:
        self.stencil = stencil
        self.h_power = h_power


class Equation2D:

    lhs_terms: list[Term2D] = []
    rhs_function: Callable = lambda x: 0

    def add_lhs_term(self, new_lhs_term: Term2D):
        self.lhs_terms.append(new_lhs_term)

    def get_discretized(self,
                        stencil_inferior_limit: int,
                        stencil_superior_limit: int,
                        convergence_order: int):
        """
        Compute the discretized version of the equation.

        Parameters
        ----------
        stencil_inferior_limit : int
            Inferior limit of the stencil.
        stencil_superior_limit : int
            superior limit of the stencil.
        convergence_order : int
            Order of convergence.

        Returns
        -------
        DiscretizedEquation2D
            Discretized version of the equation.
        """

        # Allocate a dict to store the stencils
        stencil_dict = {}

        # Iterate over the terms of the equation to discretize them
        for term in self.lhs_terms:

            # Discretize the term
            discrete_term = term.get_discretized(
                stencil_inferior_limit=stencil_superior_limit,
                stencil_superior_limit=stencil_superior_limit,
                convergence_order=convergence_order
            )

            # Create a new entry in the dict if necessary
            if not discrete_term.h_power in stencil_dict:
                stencil_dict[discrete_term.h_power] = discrete_term.stencil
            else:
                stencil_dict[discrete_term.h_power] += discrete_term.stencil

        # Create the discretized equation
        discretized_equation = DiscretizedEquation2D(
            stencil_dict, self.rhs_function)

        return discretized_equation


class DiscretizedEquation2D:

    def __init__(
            self,
            stencil_dict: dict[int, np.ndarray],
            rhs_function: Callable) -> None:
        self.stencil_dict = stencil_dict
        self.rhs_function = rhs_function


class NumericalScheme2D:

    def __init__(self) -> None:
        pass


class Problem2D:
    def __init__(self, mesh: StructuredMesh2D) -> None:
        self.mesh = mesh
        self.boundary_conditions = {}

    def define_boundary_condition(self, name: str, domain_function: Callable, value_function: Callable):
        self.boundary_conditions[name] = BoundaryCondition(
            name=name,
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
