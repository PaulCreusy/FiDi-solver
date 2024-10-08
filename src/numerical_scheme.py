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

from copy import copy
import numpy as np
from typing import Callable

# Local imports #

from .finite_differences_approx import compute_finite_difference
from .polynomials import get_monom, PolynomFraction

###########
# Classes #
###########


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

    def convert_to_numerical_scheme(self):
        """
        Convert the discrete equation into a numerical scheme.
        """

        # Create the dict of stencil terms
        stencil_terms = {}
        for h_power in self.stencil_dict:
            current_stencil = self.stencil_dict[h_power]
            for i in range(current_stencil.shape[0]):
                for j in range(current_stencil.shape[1]):
                    pos_tuple = (i, j)
                    monom = get_monom(-h_power, 1.)
                    if pos_tuple == (0, 0):
                        divider = PolynomFraction(numerator=np.array(
                            [current_stencil[i, j]]), denominator=monom)
                    else:
                        polynom_fraction = PolynomFraction(numerator=np.array(
                            [-current_stencil[i, j]]), denominator=monom)

                        if pos_tuple not in stencil_terms:
                            stencil_terms[pos_tuple] = polynom_fraction
                        else:
                            stencil_terms[pos_tuple] += polynom_fraction

        # Create the numerical scheme
        numerical_scheme = NumericalScheme2D(
            stencil_terms=stencil_terms,
            rhs_function=self.rhs_function,
            divider=divider
        )

        return numerical_scheme


class NumericalScheme2D:

    def __init__(self, stencil_terms: dict[tuple[int, int], PolynomFraction], rhs_function: Callable, divider: PolynomFraction) -> None:
        self.stencil_terms = stencil_terms
        self.rhs_function = rhs_function
        self.divider = divider
