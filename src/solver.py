"""
Module to define a simple iterative solver structure.
"""

###########
# Imports #
###########

# Python imports #

from typing import Callable

# Local imports #

from .problem import Problem2D

###########
# Classes #
###########

class Solver2D:
    def __init__(self, problem: Problem2D, discretized_equation: Callable) -> None:
        self.mesh = problem.mesh
        self.discretized_equation = discretized_equation
