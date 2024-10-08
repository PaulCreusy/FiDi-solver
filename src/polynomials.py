"""
Module to define a polynomials class.
"""

###########
# Imports #
###########

import numpy as np

#########
# Class #
#########


class Polynom:
    """
    Class to define and manipulate polynoms.
    """

    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    @property
    def order(self):
        return self.coefficients.size - 1

    def change_order(self, new_order: int):
        """
        Change the order of the polynom to the new value.

        Parameters
        ----------
        new_order : int
            New order of the polynom.
        """

        if new_order >= self.order:
            new_coefficients = np.zeros(new_order + 1, dtype=np.float64)
            new_coefficients[:self.order + 1] = self.coefficients
            self.coefficients = new_coefficients

    def evaluate(self, x: float):
        """
        Evaluate the polynom in x.

        Parameters
        ----------
        x : float
            Value for which to evaluate the polynom.

        Returns
        -------
        float
            Value of the polynom in x.
        """

        # Create an array with powers of x
        x_power = np.power(np.ones((self.order + 1,)) * x,
                           np.arange(self.order + 1))

        # Compute the value of the evaluated polynom
        res = np.sum(self.coefficients * x_power)

        return res

    def __add__(self, obj2):

        # Change the order of the two polynoms to the same one
        new_order = max(self.order, obj2.order)
        self.change_order(new_order)
        obj2.change_order(new_order)

        # Compute the new coefficients
        new_coefficients = self.coefficients + obj2.coefficients

        # Create a new polynom
        new_polynom = Polynom(new_coefficients)

        return new_polynom
