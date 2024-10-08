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

        # Create a list of empty coefficients
        new_order = max(self.order, obj2.order)
        new_coefficients = np.zeros(new_order + 1, dtype=np.float64)

        # Compute the new coefficients
        new_coefficients = self.coefficients + obj2.coefficients

        # Create a new polynom
        new_polynom = Polynom(new_coefficients)

        return new_polynom

    def __iadd__(self, obj2):
        self = self + obj2
        return self

    def __mul__(self, obj2):
        # Create a list of emtpy coefficients
        new_order = self.order + obj2.order
        new_coefficients = np.zeros(new_order + 1, dtype=np.float64)

        # Compute the new coefficients
        for i in range(self.coefficients.size):
            for j in range(obj2.coefficients.size):
                new_coefficients[i + j] = self.coefficients[i] * \
                    obj2.coefficients[j]

        # Create a new polynom
        new_polynom = Polynom(new_coefficients)

        return new_polynom

    def __eq__(self, obj2: object) -> bool:
        return (self.coefficients == obj2.coefficients).all()


class PolynomFraction:
    """
    Class to define and manipulate polynom fractions.
    """

    def __init__(self, numerator: Polynom, denominator: Polynom) -> None:
        self.numerator = numerator
        self.denominator = denominator

    def evaluate(self, x: float):
        """
        Evaluate the polynom fraction in x.

        Parameters
        ----------
        x : float
            Value for which to evaluate the polynom fraction.

        Returns
        -------
        float
            Value of the polynom fraction in x.
        """

        numerator_value = self.numerator.evaluate(x)
        denominator_value = self.denominator.evaluate(x)

        return numerator_value / denominator_value

    def __add__(self, obj2: object):

        # Compute the new numerator and denominator
        if self.denominator == obj2.denominator:
            new_numerator = self.numerator + obj2.numerator
            new_denominator = self.denominator
        else:
            new_numerator = self.numerator * obj2.denominator + \
                obj2.numerator * self.denominator
            new_denominator = self.denominator * obj2.denominator

        # Create a new polynom fraction
        new_polynom_fraction = PolynomFraction(new_numerator, new_denominator)

        return new_polynom_fraction

    def __mul__(self, obj2: object):

        # Compute the new numerator and denominator
        if isinstance(obj2, Polynom):
            new_numerator = self.numerator * obj2
            new_denominator = self.denominator
        elif isinstance(obj2, PolynomFraction):
            new_numerator = self.numerator * obj2.numerator
            new_denominator = self.denominator * obj2.denominator
        else:
            raise NotImplementedError

        # Create a new polynom fraction
        new_polynom_fraction = PolynomFraction(new_numerator, new_denominator)

        return new_polynom_fraction

    def __truediv__(self, obj2: object):

        # Compute the new numerator and denominator
        if isinstance(obj2, Polynom):
            new_denominator = self.denominator * obj2
            new_numerator = self.numerator
        elif isinstance(obj2, PolynomFraction):
            new_numerator = self.numerator * obj2.denominator
            new_denominator = self.denominator * obj2.numerator
        else:
            raise NotImplementedError

        # Create a new polynom fraction
        new_polynom_fraction = PolynomFraction(new_numerator, new_denominator)

        return new_polynom_fraction


def get_monom(order: int, factor: float):
    """
    Create the monom corresponding to factor * x ** order.

    Parameters
    ----------
    order : int
        Order of the monom.
    factor : float
        Factor of the monom.

    Returns
    -------
    Polynom
        Monom factor * x ** order.
    """

    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[-1] = factor
    polynom = Polynom(coefficients)

    return polynom
