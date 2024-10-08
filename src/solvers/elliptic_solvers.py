"""
Module defining functions to solve elliptic functions.
"""

###########
# Imports #
###########

import numpy as np
from numba import njit
from tqdm import tqdm

# Use Picard Method to linearize non linear equations


#############
# Functions #
#############

@njit
def jacobi_solver():
    sol_prec = np.zeros((Nx, Ny), dtype=np.float64)
    sol_next = np.zeros((Nx, Ny), dtype=np.float64)
    for t in tqdm(range(Nt)):
        for i in range(Nx):
            for j in range(Ny):
                if i == 0:  # Left boundary (Dirichlet)
                    sol_next[i, j] = 0.25 * rho_0 * \
                        pow(U, 2) * (1 - np.cos(2 * np.pi * y[j] / L))
                elif j == 0:  # Lower boundary (Dirichlet)
                    sol_next[i, j] = 0.25 * rho_0 * \
                        pow(U, 2) * (1 - np.cos(2 * np.pi * x[i] / L))
                elif i == Nx - 1:  # Right boundary (Neumann)
                    sol_next[i, j] = sol_prec[i - 1, j]
                elif j == Ny - 1:  # Top boundary (Neumann)
                    sol_next[i, j] = sol_prec[i, j - 1]
                else:  # Center
                    sol_next[i, j] = 0.25 * (sol_prec[i + 1, j] + sol_prec[i, j + 1] +
                                             sol_prec[i - 1, j] + sol_prec[i, j - 1]) - 0.25 * f(x[i], y[j]) * pow(h, 2)

        # Overwrite prec matrix
        sol_prec = sol_next.copy()

    return sol_next
