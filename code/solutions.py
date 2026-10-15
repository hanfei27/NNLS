"""The five exact solutions in Section 4.1 and their analytic derivatives."""
from __future__ import annotations
import math
import numpy as np

LIMITED_REGULARITY_EXPONENT = 21.0 / 4.0
LIMITED_REGULARITY_X_DECAY = 0.10
LIMITED_REGULARITY_Y_DECAY = 0.12

def _derivative_axes(alpha: tuple[int, int]) -> tuple[int, ...]:
    if len(alpha) != 2 or any(value < 0 for value in alpha):
        raise ValueError("alpha must be a nonnegative two-dimensional multi-index")
    if sum(alpha) > 3:
        raise ValueError("manufactured derivatives are implemented through order three")
    return tuple(axis for axis, count in enumerate(alpha) for _ in range(count))


def _quadratic_composition_derivative(
    value: np.ndarray,
    first_outer: np.ndarray,
    second_outer: np.ndarray,
    third_outer: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    derivative_axes: tuple[int, ...],
) -> np.ndarray:
    order = len(derivative_axes)
    if order == 0:
        return value
    i = derivative_axes[0]
    if order == 1:
        return first_outer * gradient[:, i]
    j = derivative_axes[1]
    if order == 2:
        return second_outer * gradient[:, i] * gradient[:, j] + first_outer * hessian[i, j]
    k = derivative_axes[2]
    return (
        third_outer * gradient[:, i] * gradient[:, j] * gradient[:, k]
        + second_outer
        * (
            hessian[i, j] * gradient[:, k]
            + hessian[i, k] * gradient[:, j]
            + hessian[j, k] * gradient[:, i]
        )
    )


def _gaussian_modulated_derivative(
    coordinate: np.ndarray,
    *,
    rate: float,
    frequency: complex,
    order: int,
) -> np.ndarray:
    if order not in (0, 1, 2, 3):
        raise ValueError("Gaussian modulation derivative order must be between zero and three")
    z = -rate * coordinate + frequency
    base = np.exp(-0.5 * rate * coordinate**2 + frequency * coordinate)
    polynomial = (
        np.ones_like(coordinate, dtype=complex),
        z,
        z**2 - rate,
        z**3 - 3.0 * rate * z,
    )[order]
    return polynomial * base


def _bilinear_polynomial_derivative(
    points: np.ndarray,
    alpha: tuple[int, int],
    coefficients: tuple[float, float, float, float],
) -> np.ndarray:
    constant, x_coefficient, y_coefficient, xy_coefficient = coefficients
    x = points[:, 0]
    y = points[:, 1]
    if alpha == (0, 0):
        return (
            constant
            + x_coefficient * x
            + y_coefficient * y
            + xy_coefficient * x * y
        )
    if alpha == (1, 0):
        return x_coefficient + xy_coefficient * y
    if alpha == (0, 1):
        return y_coefficient + xy_coefficient * x
    if alpha == (1, 1):
        return np.full(points.shape[0], xy_coefficient)
    return np.zeros(points.shape[0])


def _polynomial_gaussian_derivative(
    points: np.ndarray,
    alpha: tuple[int, int],
    polynomial_coefficients: tuple[float, float, float, float],
    x_rate: float,
    y_rate: float,
) -> np.ndarray:
    derivative = np.zeros(points.shape[0], dtype=float)
    for beta_x in range(alpha[0] + 1):
        for beta_y in range(alpha[1] + 1):
            beta = (beta_x, beta_y)
            coefficient = math.comb(alpha[0], beta_x) * math.comb(
                alpha[1], beta_y
            )
            polynomial = _bilinear_polynomial_derivative(
                points, beta, polynomial_coefficients
            )
            gaussian_x = _gaussian_modulated_derivative(
                points[:, 0],
                rate=x_rate,
                frequency=0.0j,
                order=alpha[0] - beta_x,
            ).real
            gaussian_y = _gaussian_modulated_derivative(
                points[:, 1],
                rate=y_rate,
                frequency=0.0j,
                order=alpha[1] - beta_y,
            ).real
            derivative += coefficient * polynomial * gaussian_x * gaussian_y
    return derivative


def _quadratic_field(
    points: np.ndarray,
    terms: tuple[tuple[float, tuple[float, float], float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.ones(points.shape[0], dtype=float)
    gradient = np.zeros((points.shape[0], 2), dtype=float)
    hessian = np.zeros((2, 2), dtype=float)
    for rate, direction_values, offset in terms:
        direction = np.asarray(direction_values, dtype=float)
        coordinate = points @ direction + offset
        value += rate * coordinate**2
        gradient += (2.0 * rate * coordinate)[:, None] * direction[None, :]
        hessian += 2.0 * rate * np.outer(direction, direction)
    return value, gradient, hessian


def _root_exponential_derivative(
    points: np.ndarray,
    derivative_axes: tuple[int, ...],
    decay: float,
    terms: tuple[tuple[float, tuple[float, float], float], ...],
) -> np.ndarray:
    q, gradient, hessian = _quadratic_field(points, terms)
    root = np.sqrt(q)
    value = np.exp(-decay * (root - 1.0))
    log_first = -decay / (2.0 * root)
    log_second = decay / (4.0 * root**3)
    log_third = -3.0 * decay / (8.0 * root**5)
    first_outer = value * log_first
    second_outer = value * (log_first**2 + log_second)
    third_outer = value * (
        log_first**3 + 3.0 * log_first * log_second + log_third
    )
    return _quadratic_composition_derivative(
        value,
        first_outer,
        second_outer,
        third_outer,
        gradient,
        hessian,
        derivative_axes,
    )


def _tanh_ridge_derivative(
    points: np.ndarray,
    alpha: tuple[int, int],
    direction_values: tuple[float, float],
    offset: float,
) -> np.ndarray:
    direction = np.asarray(direction_values, dtype=float)
    tanh = np.tanh(points @ direction + offset)
    order = sum(alpha)
    outer = (
        tanh,
        1.0 - tanh**2,
        -2.0 * tanh * (1.0 - tanh**2),
        -2.0 * (1.0 - tanh**2) * (1.0 - 3.0 * tanh**2),
    )[order]
    return outer * direction[0] ** alpha[0] * direction[1] ** alpha[1]


def _tanh_sum_derivative(
    points: np.ndarray,
    alpha: tuple[int, int],
    terms: tuple[tuple[float, tuple[float, float], float], ...],
) -> np.ndarray:
    derivative = np.zeros(points.shape[0], dtype=float)
    for coefficient, direction, offset in terms:
        derivative += coefficient * _tanh_ridge_derivative(
            points, alpha, direction, offset
        )
    return derivative


def _falling_factorial(value: float, order: int) -> float:
    result = 1.0
    for index in range(order):
        result *= value - index
    return result


def _absolute_power_derivative(
    coordinate: np.ndarray,
    exponent: float,
    order: int,
) -> np.ndarray:
    if order < 0 or order > 3:
        raise ValueError("only derivative orders zero through three are supported")
    x = np.asarray(coordinate, dtype=float)
    values = np.zeros_like(x)
    nonzero = x != 0.0
    values[nonzero] = (
        _falling_factorial(exponent, order)
        * np.abs(x[nonzero]) ** (exponent - order)
        * np.sign(x[nonzero]) ** order
    )
    return values


def _gaussian_derivative(
    coordinate: np.ndarray,
    decay: float,
    order: int,
) -> np.ndarray:
    x = np.asarray(coordinate, dtype=float)
    base = np.exp(-decay * x**2)
    if order == 0:
        return base
    if order == 1:
        return -2.0 * decay * x * base
    if order == 2:
        return (4.0 * decay**2 * x**2 - 2.0 * decay) * base
    if order == 3:
        return (-8.0 * decay**3 * x**3 + 12.0 * decay**2 * x) * base
    raise ValueError("only derivative orders zero through three are supported")


def limited_regularity_derivative(
    points: np.ndarray,
    multiindex: tuple[int, int],
) -> np.ndarray:
    """Evaluate derivatives of |x|^(21/4) exp(-0.10 x^2 - 0.12 y^2)."""

    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    x_order, y_order = multiindex
    if min(x_order, y_order) < 0 or x_order + y_order > 3:
        raise ValueError("multiindex must have total order at most three")
    x_factor = np.zeros(x.shape[0], dtype=float)
    for power_order in range(x_order + 1):
        gaussian_order = x_order - power_order
        x_factor += (
            math.comb(x_order, power_order)
            * _absolute_power_derivative(
                x[:, 0], LIMITED_REGULARITY_EXPONENT, power_order
            )
            * _gaussian_derivative(
                x[:, 0], LIMITED_REGULARITY_X_DECAY, gaussian_order
            )
        )
    y_factor = _gaussian_derivative(
        x[:, 1], LIMITED_REGULARITY_Y_DECAY, y_order
    )
    return x_factor * y_factor


def derivative(solution: str, points: np.ndarray, alpha=(0, 0)) -> np.ndarray:
    """Return D^alpha u_i, for i=1,...,5 and total order at most three."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("points must be a finite N-by-2 array")
    axes = _derivative_axes(alpha)
    if solution == "u1":
        return _polynomial_gaussian_derivative(x, alpha, (1., .03, .025, .015), .20, .24)
    if solution == "u2":
        return _root_exponential_derivative(x, axes, .50,
            ((.16, (1., .25), -.10), (.20, (-.20, 1.), .10)))
    if solution == "u3":
        return _tanh_sum_derivative(x, alpha,
            ((.70, (.14, -.08), 0.), (.30, (.08, .15), 0.)))
    if solution == "u4":
        z = .16 * x[:, 0] + .10 * x[:, 1]
        q = 1 + z*z
        outer = (np.arcsinh(z), q**-.5, -z*q**-1.5, (2*z*z-1)*q**-2.5)[sum(alpha)]
        return outer * .16**alpha[0] * .10**alpha[1]
    if solution == "u5":
        return limited_regularity_derivative(x, alpha)
    raise ValueError(f"unknown solution: {solution}")


def rhs(solution: str, points: np.ndarray) -> np.ndarray:
    """Apply -Delta + x dot grad + I analytically to the exact solution."""
    return (derivative(solution, points) - derivative(solution, points, (2, 0))
            - derivative(solution, points, (0, 2))
            + points[:, 0] * derivative(solution, points, (1, 0))
            + points[:, 1] * derivative(solution, points, (0, 1)))
