"""
Module to compute finite differences approximation.
"""

###########
# Imports #
###########

import numpy as np

#############
# Functions #
#############


def fact(n: int):
    """
    Compute the factorial of a number.

    Parameters
    ----------
    n : int
        Number for which to compute the factorial.

    Returns
    -------
    int
        Factorial of the number.
    """

    res = 1
    for i in range(1, n):
        res = res * i

    return res


def compute_taylor_approximation(order: int, diff: int = 1):
    """
    Compute the Taylor coefficients.

    Parameters
    ----------
    order : int
        Order of the Taylor approximation.
    diff : int, optional
        Distance with the point to use to compute the approximation, by default 1

    Returns
    -------
    np.ndarray
        Array containing the coefficients of the Taylor approximation.
    """

    coef_array = np.zeros(order + 1)
    for i in range(1, order + 2):
        coef_array[i - 1] = pow(diff, i - 1) / fact(i)

    return coef_array


def compute_finite_difference(target_order: int, inferior_limit: int, superior_limit: int, derivative: int = 1):
    """
    Compute a finite difference approximation for the target order with the specified stencil for the desired derivative.

    Parameters
    ----------
    target_order : int
        Target order of convergence of the approximation.
    inferior_limit : int
        Inferior limit of the stencil.
    superior_limit : int
        Superior limit of the stencil.
    derivative : int, optional
        Desired derivative, by default 1
    """

    # Allocate the Taylor table
    taylor_table = np.zeros(
        (inferior_limit + superior_limit + 2, target_order + 1), dtype=np.float64)

    # Set to 1 the coefficient of the derivative that we want to obtain
    taylor_table[0, derivative] = 1

    # Fill the table with the taylor coefficients
    for i in range(-inferior_limit, superior_limit + 1):
        if i == 0:
            # Set to 1 the coefficient of fj
            taylor_table[i + inferior_limit + 1, 0] = 1
        else:
            taylor_approximation = compute_taylor_approximation(
                target_order, diff=i)
            taylor_table[i + inferior_limit + 1, :] = taylor_approximation

    # Solve the system
    b = taylor_table[0, :]
    a = taylor_table[1:b.size + 1, :].transpose()
    solution = np.linalg.solve(a, b)

    return solution
