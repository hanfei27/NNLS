"""High-precision six-neuron inverse constants for Table 2."""
from __future__ import annotations
from dataclasses import dataclass
import math
import mpmath as mp

REPU_DEGREE = 4
WIDTH_J = 6
FORMAL_COLLISION_INDICES = (3, 4, 5, 6, 7)
FORMAL_SIGMAS = (1, 2, 3, 4)
STENCIL_BETA = (-2.5, -1.5, -.5, .5, 1.5, 2.5)
PROOF_COEFFICIENTS = tuple((-1)**i * math.comb(5, i) for i in range(6))
WORKING_DPS = 100

@dataclass(frozen=True)
class D0Parameters:
    """The centered six-atom family used in the sharpness proposition."""

    collision_index: int
    delta: mp.mpf
    biases: tuple[mp.mpf, ...]
    proof_coefficients: tuple[int, ...] = PROOF_COEFFICIENTS

    @property
    def d0_separation(self) -> mp.mpf:
        return +self.delta

    @property
    def d_pi_separation(self) -> mp.mpf:
        return +mp.pi

    @property
    def delta_pm(self) -> mp.mpf:
        return +self.delta


@dataclass(frozen=True)
class DirectGrams:
    """Exact finite derivative and Hermite-multiplier Gram matrices."""

    derivative: dict[int, mp.matrix]
    mass: mp.matrix
    sobolev: dict[int, mp.matrix]
    normalized: bool


def _normal_density(value: mp.mpf) -> mp.mpf:
    return mp.exp(-value * value / 2) / mp.sqrt(2 * mp.pi)


def _tail_moments(lower: mp.mpf, maximum_order: int) -> list[mp.mpf]:
    if maximum_order < 0:
        raise ValueError("maximum_order must be nonnegative")
    density = _normal_density(lower)
    moments = [mp.erfc(lower / mp.sqrt(2)) / 2]
    if maximum_order >= 1:
        moments.append(density)
    for order in range(2, maximum_order + 1):
        moments.append(
            lower ** (order - 1) * density
            + (order - 1) * moments[order - 2]
        )
    return moments


def _registered_mpf(value: float | mp.mpf | str) -> mp.mpf:
    return +value if isinstance(value, mp.mpf) else mp.mpf(str(value))


def build_d0_parameters(collision_index: int, *, dps: int = WORKING_DPS) -> D0Parameters:
    """Return ``delta=2^(-j)`` and the centered six registered biases."""

    if (
        isinstance(collision_index, bool)
        or not isinstance(collision_index, int)
        or collision_index not in FORMAL_COLLISION_INDICES
    ):
        raise ValueError("collision_index must be one of 3,4,5,6,7")
    if dps < 30:
        raise ValueError("dps must be at least 30")
    with mp.workdps(dps):
        delta = mp.power(2, -collision_index)
        biases = tuple(mp.mpf(str(beta)) * delta for beta in STENCIL_BETA)
        return D0Parameters(
            collision_index=collision_index,
            delta=+delta,
            biases=tuple(+bias for bias in biases),
        )


def truncated_power_inner_product(
    left_bias: float | mp.mpf | str,
    right_bias: float | mp.mpf | str,
    power: int,
    *,
    dps: int = WORKING_DPS,
) -> mp.mpf:
    """Evaluate ``int (x-bi)_+^p (x-bj)_+^p dgamma`` exactly by moments."""

    if isinstance(power, bool) or not isinstance(power, int) or power < 0:
        raise ValueError("power must be a nonnegative integer")
    if dps < 30:
        raise ValueError("dps must be at least 30")
    with mp.workdps(dps):
        left = _registered_mpf(left_bias)
        right = _registered_mpf(right_bias)
        if not mp.isfinite(left) or not mp.isfinite(right):
            raise ValueError("biases must be finite")
        moments = _tail_moments(max(left, right), 2 * power)
        value = mp.mpf(0)
        for left_power in range(power + 1):
            left_coefficient = (
                math.comb(power, left_power)
                * (-left) ** (power - left_power)
            )
            for right_power in range(power + 1):
                value += (
                    left_coefficient
                    * math.comb(power, right_power)
                    * (-right) ** (power - right_power)
                    * moments[left_power + right_power]
                )
        return +value


def stirling_derivative_weights(sigma: int) -> tuple[int, ...]:
    """Return ``S(sigma+1,r+1)``, ``r=0,...,sigma``."""

    if (
        isinstance(sigma, bool)
        or not isinstance(sigma, int)
        or sigma < 0
        or sigma > REPU_DEGREE
    ):
        raise ValueError("sigma must be an integer in [0,4]")
    size = sigma + 2
    stirling = [[0] * size for _ in range(size)]
    stirling[0][0] = 1
    for order in range(1, size):
        for blocks in range(1, order + 1):
            stirling[order][blocks] = (
                blocks * stirling[order - 1][blocks]
                + stirling[order - 1][blocks - 1]
            )
    return tuple(stirling[sigma + 1][derivative + 1] for derivative in range(sigma + 1))


def _falling_factorial(value: int, length: int) -> int:
    if length == 0:
        return 1
    return math.factorial(value) // math.factorial(value - length)


def _symmetrize(matrix: mp.matrix) -> mp.matrix:
    return (matrix + matrix.T) / 2


def assemble_direct_grams(
    parameters: D0Parameters,
    *,
    dps: int = WORKING_DPS,
    normalize: bool = True,
) -> DirectGrams:
    """Assemble ``G_r``, ``M=G_0``, and the exact integer ``K_sigma``."""

    if not isinstance(parameters, D0Parameters):
        raise TypeError("parameters must be D0Parameters")
    if dps < 30:
        raise ValueError("dps must be at least 30")
    with mp.workdps(dps):
        normalizers = tuple(
            1 / mp.sqrt(_normal_density(bias)) if normalize else mp.mpf(1)
            for bias in parameters.biases
        )
        derivative: dict[int, mp.matrix] = {}
        for derivative_order in range(REPU_DEGREE + 1):
            power = REPU_DEGREE - derivative_order
            prefactor = mp.mpf(_falling_factorial(REPU_DEGREE, derivative_order) ** 2)
            matrix = mp.zeros(WIDTH_J)
            for row in range(WIDTH_J):
                for column in range(row, WIDTH_J):
                    entry = (
                        prefactor
                        * truncated_power_inner_product(
                            parameters.biases[row],
                            parameters.biases[column],
                            power,
                            dps=dps,
                        )
                        * normalizers[row]
                        * normalizers[column]
                    )
                    matrix[row, column] = entry
                    matrix[column, row] = entry
            derivative[derivative_order] = matrix

        sobolev: dict[int, mp.matrix] = {}
        for sigma in range(REPU_DEGREE + 1):
            matrix = mp.zeros(WIDTH_J)
            for derivative_order, weight in enumerate(
                stirling_derivative_weights(sigma)
            ):
                matrix += weight * derivative[derivative_order]
            sobolev[sigma] = _symmetrize(matrix)
        return DirectGrams(
            derivative=derivative,
            mass=derivative[0],
            sobolev=sobolev,
            normalized=normalize,
        )


def _matrix_infinity_norm(matrix: mp.matrix) -> mp.mpf:
    return max(
        sum(abs(matrix[row, column]) for column in range(matrix.cols))
        for row in range(matrix.rows)
    )


def _vector_infinity_norm(vector: mp.matrix) -> mp.mpf:
    return max(abs(vector[index]) for index in range(vector.rows))


def _quadratic_form(matrix: mp.matrix, vector: mp.matrix) -> mp.mpf:
    return (vector.T * matrix * vector)[0]


def generalized_metrics(
    grams: DirectGrams,
    sigma: int,
    *,
    parameters: D0Parameters,
    dps: int = WORKING_DPS,
) -> dict[str, mp.mpf | int]:
    """Compute the maximal pencil value and the fixed-proof-vector quotient."""

    if sigma not in FORMAL_SIGMAS:
        raise ValueError("sigma must be one of 1,2,3,4")
    if dps < 30:
        raise ValueError("dps must be at least 30")
    with mp.workdps(dps):
        mass = _symmetrize(grams.mass)
        sobolev = _symmetrize(grams.sobolev[sigma])
        mass_values = mp.eigsy(mass, eigvals_only=True)
        mass_maximum = mass_values[mass_values.rows - 1]
        rank_threshold = WIDTH_J * mp.eps * mass_maximum
        rank = sum(value > rank_threshold for value in mass_values)
        if rank != WIDTH_J or mass_values[0] <= 0:
            raise RuntimeError(
                f"direct mass Gram is not positive definite for j={parameters.collision_index}"
            )

        cholesky = mp.cholesky(mass)
        inverse_cholesky = cholesky**-1
        whitened = _symmetrize(inverse_cholesky * sobolev * inverse_cholesky.T)
        values, vectors = mp.eigsy(whitened)
        largest = values.rows - 1
        generalized_value = values[largest]
        if generalized_value <= 0:
            raise RuntimeError("largest generalized eigenvalue must be positive")
        whitened_vector = vectors[:, largest]
        generalized_vector = inverse_cholesky.T * whitened_vector
        residual = sobolev * generalized_vector - generalized_value * mass * generalized_vector
        residual_scale = (
            _matrix_infinity_norm(sobolev)
            + generalized_value * _matrix_infinity_norm(mass)
        ) * _vector_infinity_norm(generalized_vector)
        if residual_scale <= 0:
            raise RuntimeError("generalized eigenpair residual has zero scale")
        relative_residual = _vector_infinity_norm(residual) / residual_scale

        if grams.normalized:
            proof_vector = mp.matrix(
                [
                    mp.sqrt(_normal_density(bias)) * coefficient
                    for bias, coefficient in zip(
                        parameters.biases,
                        parameters.proof_coefficients,
                        strict=True,
                    )
                ]
            )
        else:
            proof_vector = mp.matrix(parameters.proof_coefficients)
        proof_denominator = _quadratic_form(mass, proof_vector)
        proof_numerator = _quadratic_form(sobolev, proof_vector)
        if proof_denominator <= 0 or proof_numerator <= 0:
            raise RuntimeError("fixed proof-vector quotient must be positive")
        proof_quotient = proof_numerator / proof_denominator
        return {
            "mass_rank": int(rank),
            "mass_rank_threshold": +rank_threshold,
            "mass_lambda_min": +mass_values[0],
            "mass_lambda_max": +mass_maximum,
            "mass_condition": +(mass_maximum / mass_values[0]),
            "generalized_lambda_max": +generalized_value,
            "inverse_norm_max": +mp.sqrt(generalized_value),
            "generalized_eigen_residual": +relative_residual,
            "proof_generalized_rayleigh": +proof_quotient,
            "proof_norm_ratio": +mp.sqrt(proof_quotient),
        }


