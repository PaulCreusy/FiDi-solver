"""
Module to test the solver on the Poisson equation.
"""

###########
# Imports #
###########

# Python imports #

import sys
import numpy as np
from numba import jit

# Set the path for local imports
sys.path.append(".")

# Local imports #

from src.problem import Term2D, Equation2D, DiscretizedEquation2D

#############
# Functions #
#############


@jit(nopython=True)
def rhs(x, y):
    res = rho_0 * pow((np.pi * U) / L, 2) * \
        (np.cos(2 * np.pi * x / L) + np.cos(2 * np.pi * y / L))
    return res

##################
# Test variables #
##################


# Define the physical parameters
rho_0 = 1.2
U = 0.1
L = 1

# Define the pressure term
p = Term2D()

# Create the x derivative
p_x = p.dx()
p_xx = p_x.dx()

# Create the y derivative
p_y = p.dy()
p_yy = p_y.dy()

# Create the equation p_xx + p_yy = rhs(x,y)
equation = Equation2D()
equation.add_lhs_term(p_xx)
equation.add_lhs_term(p_yy)
equation.rhs_function = rhs

# Discretize the equation
discrete_equation = equation.get_discretized(
    stencil_inferior_limit=1,
    stencil_superior_limit=1,
    convergence_order=2
)

# Define the boundary conditions

# Create the mesh

# Define the problem

# Solve the problem
