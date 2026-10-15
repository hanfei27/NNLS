"""ReLU^4 discretization, Gaussian quadrature, and coefficient solvers."""
from __future__ import annotations
from dataclasses import dataclass
from math import factorial, gamma, pi
from typing import Iterator
import math
import numpy as np
from scipy.linalg import lstsq
from solutions import derivative, rhs

DEFAULT_STREAM_CHUNK_POINTS = 4096

@dataclass(frozen=True)
class RidgeParameters:
    """Frozen inner RePU parameters theta_j = (a_j, b_j)."""

    directions: np.ndarray
    biases: np.ndarray

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions, dtype=float)
        biases = np.asarray(self.biases, dtype=float)
        if directions.ndim != 2:
            raise ValueError("directions must have shape (n_atoms, dimension)")
        if biases.ndim != 1:
            raise ValueError("biases must have shape (n_atoms,)")
        if directions.shape[0] != biases.shape[0]:
            raise ValueError("directions and biases must have the same number of atoms")
        if directions.shape[0] == 0:
            raise ValueError("at least one ridge atom is required")
        if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(biases)):
            raise ValueError("ridge parameters must be finite")
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("ridge directions must be unit vectors")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "biases", biases)

    @property
    def n_atoms(self) -> int:
        return int(self.biases.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.directions.shape[1])


def as_points(points: np.ndarray, dimension: int) -> np.ndarray:
    """Return points as an array with shape (n_points, dimension)."""

    array = np.asarray(points, dtype=float)
    if array.ndim == 1 and dimension == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"points must have shape (n_points, {dimension})")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must be finite")
    return array


def ridge_inputs(points: np.ndarray, params: RidgeParameters) -> np.ndarray:
    x = as_points(points, params.dimension)
    return x @ params.directions.T - params.biases[None, :]


def repu_derivative_matrix(
    points: np.ndarray,
    params: RidgeParameters,
    k: int,
    alpha: tuple[int, ...],
) -> np.ndarray:
    """Evaluate one weak partial derivative of all RePU atoms."""

    if k < 1:
        raise ValueError("RePU degree k must be at least 1")
    if len(alpha) != params.dimension or any(value < 0 for value in alpha):
        raise ValueError("alpha must be a nonnegative multi-index matching the dimension")
    order = int(sum(alpha))
    x = as_points(points, params.dimension)
    if order > k:
        return np.zeros((x.shape[0], params.n_atoms), dtype=float)
    inputs = ridge_inputs(x, params)
    positive = np.maximum(inputs, 0.0)
    active_power = np.where(inputs > 0.0, positive ** (k - order), 0.0)
    falling = factorial(k) / factorial(k - order)
    direction_factor = np.prod(
        params.directions ** np.asarray(alpha, dtype=int)[None, :],
        axis=1,
    )
    return falling * active_power * direction_factor[None, :]


def ornstein_uhlenbeck_residual_matrix(
    points: np.ndarray,
    params: RidgeParameters,
    k: int,
) -> np.ndarray:
    """Evaluate ``(-Delta + x dot grad + I) phi_j`` for all ridge atoms."""

    x = as_points(points, params.dimension)
    values = repu_derivative_matrix(x, params, k, (0,) * params.dimension)
    laplacian = np.zeros_like(values)
    gradient_term = np.zeros_like(values)
    for coordinate in range(params.dimension):
        second_alpha = tuple(
            2 if axis == coordinate else 0 for axis in range(params.dimension)
        )
        first_alpha = tuple(
            1 if axis == coordinate else 0 for axis in range(params.dimension)
        )
        laplacian += repu_derivative_matrix(x, params, k, second_alpha)
        gradient_term += x[:, coordinate, None] * repu_derivative_matrix(
            x,
            params,
            k,
            first_alpha,
        )
    return -laplacian + gradient_term + values


def registered_ridge_parameters(
    *,
    n_directions: int,
    n_bias: int,
    bias_radius: float,
    bias_shift: float,
) -> RidgeParameters:
    """Build the locked equal-angle, shifted-uniform tensor dictionary."""

    if n_directions < 3:
        raise ValueError("n_directions must be at least three")
    if n_bias < 2:
        raise ValueError("n_bias must be at least two")
    if bias_radius <= 0.0:
        raise ValueError("bias_radius must be positive")
    if not 0.0 < bias_shift < 1.0:
        raise ValueError("bias_shift must lie strictly between zero and one")
    angles = 2.0 * np.pi * np.arange(n_directions, dtype=float) / n_directions
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    step = 2.0 * bias_radius / n_bias
    biases = -bias_radius + (np.arange(n_bias, dtype=float) + bias_shift) * step
    return RidgeParameters(
        directions=np.repeat(directions, n_bias, axis=0),
        biases=np.tile(biases, n_directions),
    )


def parameter_geometry(
    params: RidgeParameters,
    *,
    n_bias: int,
    bias_radius: float,
) -> dict[str, float]:
    """Return exact fill/separation diagnostics for a tensor parameter set."""

    if params.dimension != 2:
        raise ValueError("registered parameter geometry requires dimension two")
    if n_bias < 2 or params.n_atoms % n_bias != 0:
        raise ValueError("n_bias must define a nontrivial tensor parameter set")
    if bias_radius <= 0.0:
        raise ValueError("bias_radius must be positive")
    n_directions = params.n_atoms // n_bias
    directions = params.directions[::n_bias]
    biases = np.sort(params.biases[:n_bias])
    expected_directions = np.repeat(directions, n_bias, axis=0)
    expected_biases = np.tile(params.biases[:n_bias], n_directions)
    if not np.allclose(params.directions, expected_directions, rtol=0.0, atol=1e-13):
        raise ValueError("directions do not have tensor-product ordering")
    if not np.allclose(params.biases, expected_biases, rtol=0.0, atol=1e-13):
        raise ValueError("biases do not have tensor-product ordering")

    angles = np.sort(np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi))
    angular_gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    direction_fill = 0.5 * float(np.max(angular_gaps))
    direction_separation = float(np.min(angular_gaps))
    bias_gaps = np.diff(biases)
    bias_fill = max(
        float(biases[0] + bias_radius),
        float(bias_radius - biases[-1]),
        0.5 * float(np.max(bias_gaps)),
    )
    bias_separation = float(np.min(bias_gaps))
    h_theta = direction_fill + bias_fill
    delta = min(direction_separation, bias_separation)

    direction_antipodal = float("inf")
    for first in directions:
        dots = np.clip(directions @ (-first), -1.0, 1.0)
        direction_antipodal = min(direction_antipodal, float(np.min(np.arccos(dots))))
    bias_antipodal = float(np.min(np.abs(biases[:, None] + biases[None, :])))
    d_pi_separation = direction_antipodal + bias_antipodal
    delta_pm = min(delta, d_pi_separation)
    if delta <= 0.0:
        raise RuntimeError("ordinary parameter separation is not positive")

    return {
        "h_theta": h_theta,
        "delta": delta,
        "delta_pm": delta_pm,
        "direction_fill": direction_fill,
        "bias_fill": bias_fill,
        "direction_separation": direction_separation,
        "bias_separation": bias_separation,
        "direction_antipodal_separation": direction_antipodal,
        "bias_antipodal_separation": bias_antipodal,
        "d_pi_separation": d_pi_separation,
        "h_theta_over_delta": h_theta / delta,
        "h_theta_over_delta_pm": float("inf") if delta_pm == 0.0 else h_theta / delta_pm,
    }


def gaussian_density(points: np.ndarray) -> np.ndarray:
    """Standard d-dimensional Gaussian density rho_gamma(x)."""

    x = as_points(points, 2)
    dimension = x.shape[1]
    normalizer = (2.0 * np.pi) ** (-0.5 * dimension)
    return normalizer * np.exp(-0.5 * np.sum(x * x, axis=1))


def sample_uniform_ball(
    *,
    n_samples: int,
    radius: float,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Draw iid points from the Euclidean uniform law on ``B_R``."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_samples, dimension))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        zero = norms == 0.0
        directions[zero] = rng.normal(size=(int(np.count_nonzero(zero)), dimension))
        norms = np.linalg.norm(directions, axis=1)
    directions /= norms[:, None]
    radii = radius * rng.random(n_samples) ** (1.0 / dimension)
    return directions * radii[:, None]


def angles_to_directions(angles):
    return np.column_stack([np.cos(angles), np.sin(angles)])


def shifted_uniform_biases(n_bias, bias_radius, shift):
    return -bias_radius + (np.arange(n_bias, dtype=float) + shift) * (2 * bias_radius / n_bias)


def random_iid_parameters(
    *,
    n_atoms: int,
    dimension: int,
    bias_radius: float,
    seed: int,
) -> RidgeParameters:
    """Generate iid random shallow-network parameters.

    Directions are unit vectors.  Biases are sampled in the physical interval
    ``[-bias_radius, bias_radius]`` and are not normalized.
    """

    if n_atoms < 1:
        raise ValueError("n_atoms must be positive")
    if dimension != 2:
        raise ValueError("random_iid_parameters is implemented for dimension=2")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, n_atoms)
    directions = angles_to_directions(angles)
    biases = rng.uniform(-bias_radius, bias_radius, n_atoms)
    return RidgeParameters(directions=directions, biases=biases)


def random_tensor_parameters(
    *,
    n_directions: int,
    n_bias: int,
    bias_radius: float,
    seed: int,
    stratified: bool,
    bias_shift: float,
) -> RidgeParameters:
    """Generate random directions crossed with the same shifted uniform bias grid."""

    if n_directions < 3:
        raise ValueError("n_directions must be at least 3")
    rng = np.random.default_rng(seed)
    if stratified:
        angles = 2.0 * np.pi * (np.arange(n_directions) + rng.uniform(0.0, 1.0, n_directions)) / n_directions
    else:
        angles = rng.uniform(0.0, 2.0 * np.pi, n_directions)
    directions = angles_to_directions(np.sort(np.mod(angles, 2.0 * np.pi)))
    bias_grid = shifted_uniform_biases(n_bias, bias_radius, bias_shift)
    full_directions = np.repeat(directions, n_bias, axis=0)
    full_biases = np.tile(bias_grid, n_directions)
    return RidgeParameters(directions=full_directions, biases=full_biases)


@dataclass(frozen=True)
class SolverResult:
    coefficients: np.ndarray
    l1_norm: float
    split_l1_norm: float
    weighted_residual_norm: float
    objective_value: float
    iterations: int
    success: bool
    message: str


def validate_least_squares_inputs(
    matrix: np.ndarray,
    rhs: np.ndarray,
    weights: np.ndarray,
    l1_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bmat = np.asarray(matrix, dtype=float)
    y = np.asarray(rhs, dtype=float)
    w = np.asarray(weights, dtype=float)
    if bmat.ndim != 2:
        raise ValueError("matrix must have shape (n_samples, n_atoms)")
    if y.shape != (bmat.shape[0],):
        raise ValueError("rhs must have shape (n_samples,)")
    if w.shape != (bmat.shape[0],):
        raise ValueError("weights must have shape (n_samples,)")
    if l1_radius <= 0.0:
        raise ValueError("l1_radius must be positive")
    if not np.all(np.isfinite(bmat)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)):
        raise ValueError("matrix, rhs, and weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("weights must be nonnegative")
    return bmat, y, w


def weighted_design(
    matrix: np.ndarray,
    rhs: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_w = np.sqrt(weights)
    return matrix * sqrt_w[:, None], rhs * sqrt_w


def l1_kkt_diagnostics(
    matrix: np.ndarray,
    rhs: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    l1_radius: float,
    *,
    feasibility_tol: float = 1e-7,
    support_tol: float = 1e-10,
) -> dict[str, float]:
    """Evaluate primal, stationarity, subgradient, and complementarity residuals."""

    bmat = np.asarray(matrix, dtype=float)
    y = np.asarray(rhs, dtype=float)
    w = np.asarray(weights, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if bmat.ndim != 2 or y.shape != (bmat.shape[0],):
        raise ValueError("matrix and rhs shapes are inconsistent")
    if w.shape != (bmat.shape[0],) or c.shape != (bmat.shape[1],):
        raise ValueError("weights or coefficients have inconsistent shape")
    if l1_radius <= 0.0 or feasibility_tol <= 0.0 or support_tol <= 0.0:
        raise ValueError("radii and tolerances must be positive")
    if np.any(w < 0.0) or not all(
        np.all(np.isfinite(array)) for array in (bmat, y, w, c)
    ):
        raise ValueError("KKT inputs must be finite and weights nonnegative")

    weighted_matrix, weighted_rhs = weighted_design(bmat, y, w)
    gradient = weighted_matrix.T @ (weighted_matrix @ c - weighted_rhs)
    l1_norm = float(np.sum(np.abs(c)))
    primal_violation = max(0.0, l1_norm - l1_radius)
    support_threshold = support_tol * max(1.0, float(np.max(np.abs(c), initial=0.0)))
    support = np.abs(c) > support_threshold
    constraint_active = l1_norm >= l1_radius - feasibility_tol
    if constraint_active:
        if not np.any(support):
            raise RuntimeError("active l1 constraint has empty numerical support")
        multiplier_candidates = -gradient[support] * np.sign(c[support])
        dual_lambda = max(0.0, float(np.median(multiplier_candidates)))
    else:
        dual_lambda = 0.0

    support_residual = (
        float(np.max(np.abs(gradient[support] + dual_lambda * np.sign(c[support]))))
        if np.any(support)
        else 0.0
    )
    zero_violation = (
        float(np.max(np.maximum(np.abs(gradient[~support]) - dual_lambda, 0.0)))
        if np.any(~support)
        else 0.0
    )
    stationarity = max(support_residual, zero_violation)
    stationarity_scale = max(1.0, float(np.max(np.abs(weighted_matrix.T @ weighted_rhs))))
    complementarity = abs(dual_lambda * (l1_radius - l1_norm))
    return {
        "l1_norm": l1_norm,
        "primal_violation": float(primal_violation),
        "dual_lambda": float(dual_lambda),
        "support_stationarity_residual": support_residual,
        "zero_subgradient_violation": zero_violation,
        "stationarity_residual": stationarity,
        "relative_stationarity_residual": float(stationarity / stationarity_scale),
        "complementarity_residual": float(complementarity),
        "constraint_active": float(constraint_active),
    }


def project_onto_l1_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    """Return the Euclidean projection of a vector onto an l1 ball."""

    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("vector must be finite and one-dimensional")
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    magnitudes = np.abs(values)
    if float(np.sum(magnitudes)) <= radius:
        return values.copy()
    ordered = np.sort(magnitudes)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, ordered.size + 1, dtype=float)
    active = np.flatnonzero(ordered * indices > cumulative - radius)
    if active.size == 0:
        raise RuntimeError("l1-ball projection could not determine its active set")
    last = int(active[-1])
    threshold = (cumulative[last] - radius) / float(last + 1)
    projected = np.sign(values) * np.maximum(magnitudes - threshold, 0.0)
    if float(np.sum(np.abs(projected))) > radius + 1.0e-10 * max(1.0, radius):
        raise RuntimeError("l1-ball projection returned an infeasible vector")
    return projected


def solve_l1_least_squares_projected(
    matrix: np.ndarray,
    rhs: np.ndarray,
    weights: np.ndarray,
    l1_radius: float,
    *,
    maxiter: int = 250_000,
    kkt_tol: float = 5.0e-5,
    check_interval: int = 500,
    feasibility_tol: float = 1.0e-7,
) -> SolverResult:
    """Solve constrained least squares by FISTA on the l1 ball."""

    if maxiter < 1 or check_interval < 1:
        raise ValueError("iteration limits must be positive")
    if kkt_tol <= 0.0 or feasibility_tol <= 0.0:
        raise ValueError("solver tolerances must be positive")
    bmat, y, w = validate_least_squares_inputs(matrix, rhs, weights, l1_radius)
    weighted_matrix, weighted_rhs = weighted_design(bmat, y, w)
    unconstrained = np.linalg.lstsq(weighted_matrix, weighted_rhs, rcond=None)[0]
    unconstrained_l1 = float(np.sum(np.abs(unconstrained)))
    if unconstrained_l1 <= l1_radius + feasibility_tol:
        residual = weighted_matrix @ unconstrained - weighted_rhs
        residual_norm = float(np.linalg.norm(residual))
        return SolverResult(
            coefficients=unconstrained,
            l1_norm=unconstrained_l1,
            split_l1_norm=unconstrained_l1,
            weighted_residual_norm=residual_norm,
            objective_value=0.5 * residual_norm**2,
            iterations=0,
            success=True,
            message="unconstrained solution is feasible",
        )

    gram = weighted_matrix.T @ weighted_matrix
    correlation = weighted_matrix.T @ weighted_rhs
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(lipschitz) or lipschitz <= 0.0:
        raise RuntimeError("weighted design has no positive finite spectral norm")

    coefficients = project_onto_l1_ball(unconstrained, l1_radius)
    extrapolated = coefficients.copy()
    momentum = 1.0
    final_diagnostics: dict[str, float] | None = None
    final_iteration = 0
    for iteration in range(1, maxiter + 1):
        gradient = gram @ extrapolated - correlation
        updated = project_onto_l1_ball(
            extrapolated - gradient / lipschitz,
            l1_radius,
        )
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        extrapolated = updated + ((momentum - 1.0) / next_momentum) * (
            updated - coefficients
        )
        coefficients = updated
        momentum = float(next_momentum)
        if iteration % check_interval == 0 or iteration == maxiter:
            final_diagnostics = l1_kkt_diagnostics(
                bmat,
                y,
                w,
                coefficients,
                l1_radius,
                feasibility_tol=feasibility_tol,
            )
            if (
                final_diagnostics["primal_violation"] <= feasibility_tol
                and final_diagnostics["relative_stationarity_residual"] <= kkt_tol
            ):
                final_iteration = iteration
                break
    if final_iteration == 0 or final_diagnostics is None:
        residual = (
            float("nan")
            if final_diagnostics is None
            else final_diagnostics["relative_stationarity_residual"]
        )
        raise RuntimeError(
            "projected FISTA did not satisfy the KKT tolerance after "
            f"{maxiter} iterations: relative residual={residual:.6g}"
        )
    residual = weighted_matrix @ coefficients - weighted_rhs
    residual_norm = float(np.linalg.norm(residual))
    if not np.isfinite(residual_norm):
        raise RuntimeError("weighted residual norm is not finite")
    l1_norm = float(np.sum(np.abs(coefficients)))
    return SolverResult(
        coefficients=coefficients,
        l1_norm=l1_norm,
        split_l1_norm=l1_norm,
        weighted_residual_norm=residual_norm,
        objective_value=0.5 * residual_norm**2,
        iterations=final_iteration,
        success=True,
        message="projected FISTA satisfied the KKT tolerance",
    )


def shared_qr_solve(
    weighted_matrix: np.ndarray,
    weighted_rhs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Solve all right-hand sides through one pivoted-QR LAPACK call."""

    matrix = np.asarray(weighted_matrix, dtype=float)
    rhs = np.asarray(weighted_rhs, dtype=float)
    if (
        matrix.ndim != 2
        or rhs.ndim != 2
        or matrix.shape[0] != rhs.shape[0]
        or rhs.shape[1] == 0
    ):
        raise ValueError(
            "expected an N x J matrix and nonempty N x S RHS with matching N"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("weighted matrix and RHS must be finite")

    coefficients, _, rank, _ = lstsq(
        matrix,
        rhs,
        cond=None,
        overwrite_a=False,
        overwrite_b=False,
        check_finite=False,
        lapack_driver="gelsy",
    )
    return np.asarray(coefficients, dtype=float), int(rank)


def _composite_gauss_1d(
    *,
    radius: float,
    cell_width: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("radius must be positive and finite")
    if cell_width <= 0.0 or not math.isfinite(cell_width):
        raise ValueError("cell_width must be positive and finite")
    if order < 1:
        raise ValueError("order must be positive")
    cell_count_float = 2.0 * radius / cell_width
    cell_count = int(round(cell_count_float))
    if cell_count <= 0 or not math.isclose(
        cell_count_float,
        cell_count,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("cell_width must partition [-radius,radius]")
    if cell_count % 2:
        raise ValueError("x=0 must be a composite-cell boundary")

    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(order)
    left_edges = -radius + cell_width * np.arange(cell_count)
    centers = left_edges + 0.5 * cell_width
    nodes = (
        centers[:, None] + 0.5 * cell_width * reference_nodes[None, :]
    ).reshape(-1)
    weights = (
        np.broadcast_to(
            0.5 * cell_width * reference_weights[None, :],
            (cell_count, order),
        ).reshape(-1)
        * np.exp(-0.5 * nodes**2)
        / math.sqrt(2.0 * math.pi)
    )
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise RuntimeError("limited quadrature weights must be positive and finite")
    return nodes, weights


def iter_limited_quadrature_chunks(
    *,
    radius: float,
    cell_width: float,
    order: int,
    max_points: int = DEFAULT_STREAM_CHUNK_POINTS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield the tensor composite rule without materializing its full point set."""

    if max_points < 1:
        raise ValueError("max_points must be positive")
    nodes, weights = _composite_gauss_1d(
        radius=radius,
        cell_width=cell_width,
        order=order,
    )
    for x_value, x_weight in zip(nodes, weights):
        for start in range(0, nodes.size, max_points):
            stop = min(start + max_points, nodes.size)
            block_size = stop - start
            points = np.empty((block_size, 2), dtype=float)
            points[:, 0] = x_value
            points[:, 1] = nodes[start:stop]
            yield points, x_weight * weights[start:stop]


def parameters(n_directions: int, n_bias: int, shift: float = .75) -> RidgeParameters:
    return registered_ridge_parameters(n_directions=n_directions, n_bias=n_bias,
                                      bias_radius=5., bias_shift=shift)


def residual_rule(cell_width: float, rotation: float, radius: float = 5.5):
    """The original rotated order-three tensor rule, clipped to the disk."""
    if cell_width <= 0 or radius <= 0:
        raise ValueError("cell width and radius must be positive")
    limit = int(np.ceil(2 * radius / cell_width)) + 2
    indices = np.arange(-limit, limit + 1, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    axis = ((indices[:, None] + .5 * (nodes + 1)[None, :]) * cell_width).ravel()
    axis_weights = np.tile(.5 * weights * cell_width, indices.size)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    wx, wy = np.meshgrid(axis_weights, axis_weights, indexing="ij")
    c, s = float(np.cos(rotation)), float(np.sin(rotation))
    points = np.column_stack([x.ravel(), y.ravel()]) @ np.array([[c, -s], [s, c]]).T
    inside = np.sum(points**2, axis=1) <= radius**2 + 1.e-13
    points, areas = points[inside], (wx * wy).ravel()[inside]
    order = np.lexsort((points[:, 1], points[:, 0]))
    points, areas = points[order], areas[order]
    return points, areas * gaussian_density(points)


def level_rule(n_directions: int, n_bias: int, factor: float = 3.):
    params = parameters(n_directions, n_bias)
    delta = parameter_geometry(params, n_bias=n_bias, bias_radius=5.)["delta_pm"]
    return residual_rule(factor * delta, math.pi / n_directions)


def solve(params, points, weights, names, radii=None, *, random_protocol=False):
    """Pivoted QR, or the original constrained random-sampling protocol.

    The latter retains the historical SVD minimum-norm initialization before
    projected FISTA. Changing that initialization can change the chosen
    representative in a rank-deficient trial space.
    """
    matrix = ornstein_uhlenbeck_residual_matrix(points, params, 4)
    targets = np.column_stack([rhs(name, points) for name in names])
    weighted = matrix * np.sqrt(weights)[:, None]
    coefficients, _ = shared_qr_solve(weighted, targets * np.sqrt(weights)[:, None])
    info = []
    for i, name in enumerate(names):
        radius = None if radii is None else radii[i]
        iterations = 0
        method = "pivoted_qr"
        if radius is not None and (random_protocol or np.abs(coefficients[:, i]).sum() > radius):
            result = solve_l1_least_squares_projected(matrix, targets[:, i], weights, radius,
                maxiter=500000, kkt_tol=1.e-8 if random_protocol else 1.e-7,
                check_interval=100 if random_protocol else 1000)
            coefficients[:, i] = result.coefficients
            iterations = result.iterations
            method = "projected_fista" if iterations else "svd_minimum_norm"
        row = {"solver": method, "iterations": iterations,
               "l1_norm": float(np.abs(coefficients[:, i]).sum())}
        if radius is not None:
            diagnostics = l1_kkt_diagnostics(matrix, targets[:, i], weights,
                                           coefficients[:, i], radius)
            row["relative_kkt_stationarity"] = diagnostics["relative_stationarity_residual"]
            row["kkt_primal_violation"] = diagnostics["primal_violation"]
            if row["kkt_primal_violation"] > 1.e-7 or row["relative_kkt_stationarity"] > 1.e-7:
                raise RuntimeError(f"coefficient solve failed KKT checks: {row}")
        info.append(row)
    return coefficients, weighted, info


def spectrum(weighted_matrix):
    """Rank and capped condition number of the column-normalized matrix."""
    norms = np.linalg.norm(weighted_matrix, axis=0)
    normalized = np.zeros_like(weighted_matrix)
    nonzero = norms > 0
    normalized[:, nonzero] = weighted_matrix[:, nonzero] / norms[nonzero]
    singular = np.linalg.svd(normalized, compute_uv=False)
    threshold = max(normalized.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > threshold * singular[0]))
    width = normalized.shape[1]
    condition = 1 / threshold if rank < width else min(singular[0] / singular[-1], 1 / threshold)
    return {"rank": rank, "condition": float(condition)}


def errors(params, coefficients, names, *, radius=8., cell_width=.05, order=4):
    """Relative H_gamma^s derivative errors, s=0,...,3, on [-radius,radius]^2."""
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (params.n_atoms, len(names)):
        raise ValueError("coefficient shape must match the dictionary and solutions")
    exact_sq = np.zeros((4, len(names)))
    error_sq = np.zeros_like(exact_sq)
    for points, weights in iter_limited_quadrature_chunks(
            radius=radius, cell_width=cell_width, order=order):
        positive = np.maximum(ridge_inputs(points, params), 0.)
        for total in range(4):
            power = positive ** (4 - total)
            for x_order in range(total + 1):
                alpha = (x_order, total - x_order)
                factors = (factorial(4) / factorial(4-total)
                    * params.directions[:, 0]**alpha[0] * params.directions[:, 1]**alpha[1])
                approx = power @ (factors[:, None] * coefficients)
                exact = np.column_stack([derivative(name, points, alpha) for name in names])
                exact_sq[total] += np.sum(weights[:, None] * exact**2, axis=0)
                error_sq[total] += np.sum(weights[:, None] * (approx - exact)**2, axis=0)
    return np.sqrt(np.cumsum(error_sq, axis=0) / np.cumsum(exact_sq, axis=0)).T
