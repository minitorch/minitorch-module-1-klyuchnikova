"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, TypeVar, Union

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value."""
    return abs(x - y) < tol


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    """Applies the ReLU activation function."""
    return max(0.0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function."""
    return math.exp(x + 1e-9)


def inv(x: float) -> float:
    """Calculates the reciprocal."""
    return 1.0 / (x + 1e-9)


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log times a second arg."""
    return d / (x + 1e-9)


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg."""
    return -d / (x**2)


def relu_back(x: Union[float, bool], d: float) -> float:
    """Computes the derivative of ReLU times a second arg."""
    x = x if isinstance(x, bool) else x > 0
    return d if x else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates all elements in a list using map."""
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two lists using zipWith."""
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sums all elements in a list using reduce."""
    return reduce(add, lst, 0.0)


def prod(lst: Iterable[float]) -> float:
    """Calculates the product of all elements in a list using reduce."""
    return reduce(mul, lst, 1.0)


U = TypeVar("U")
T = TypeVar("T")


def map(fn: Callable[[T], U], iterable: Iterable[T]) -> Iterable[U]:
    """Applies a given function to each element of an iterable."""
    return [fn(x) for x in iterable]


def zipWith(
    fn: Callable[[T, T], U], iter1: Iterable[T], iter2: Iterable[T]
) -> Iterable[U]:
    """Combines elements from two iterables using a given function."""
    return [fn(x, y) for x, y in zip(iter1, iter2)]


def reduce(fn: Callable[[U, U], U], iterable: Iterable[U], initializer: U) -> U:
    """Reduces an iterable to a single value using a given function."""
    it = iter(iterable)
    if initializer is None:
        try:
            initializer = next(it)
        except StopIteration:
            raise TypeError("reduce failed: empty sequence with no initial value")
    accum_value = initializer
    for x in it:
        accum_value = fn(accum_value, x)
    return accum_value