"""Two-group model covariance from origin-centered enclosing ellipsoids.

Alternatives within a group are bounded templates, not independent Gaussian
nuisances. Each group's unit ellipsoid contains every supplied signed shift.
Adding the two groups is the adopted independent-group quadrature convention;
it is not a simultaneous worst-case bound or a calibrated confidence region.
"""

import numpy as np
from scipy.optimize import minimize


MODEL_GROUPS = {
    "parton_shower": ("model_vincia", "fsrUp", "fsrDown"),
    "hadronization": ("model_cr1", "model_cr2", "model_fraghard", "model_fragsoft"),
}


def enclosing_template_covariance(shifts, *, rank_rtol=1e-12):
    """Return covariance and containment diagnostics for bin-by-template shifts.

    Fit only in the numerical template span. In whitened coordinates Z, the
    dual maximizes logdet(Z diag(weights) Z.T) over the weight simplex.
    Multiplication by the span rank gives the centered enclosing ellipsoid.
    No physical-space diagonal regularization is added.
    """
    shifts = np.asarray(shifts, dtype=float)
    if shifts.ndim != 2 or shifts.shape[1] == 0 or not np.isfinite(shifts).all():
        raise ValueError("Expected finite bin-by-template shifts with at least one template")
    basis, singular_values, coordinates = np.linalg.svd(shifts, full_matrices=False)
    rank = int(np.sum(singular_values > rank_rtol * singular_values[0])) if singular_values.size else 0
    if rank == 0:
        return np.zeros((shifts.shape[0], shifts.shape[0])), {
            "rank": 0, "weights": [0.0] * shifts.shape[1],
            "mahalanobis_squared": [0.0] * shifts.shape[1],
            "span_relative_residual": 0.0, "rank_rtol": rank_rtol,
            "optimality_gap": 0.0, "containment_inflation": 1.0,
        }
    transform = basis[:, :rank] * singular_values[:rank]
    coordinates = coordinates[:rank]
    residual = np.linalg.norm(shifts - transform @ coordinates) / np.linalg.norm(shifts)
    if residual > 10 * rank_rtol:
        raise ValueError("Template span truncation discarded a non-negligible direction")

    def objective(weights):
        moment = (coordinates * weights) @ coordinates.T
        sign, logdet = np.linalg.slogdet(moment)
        if sign <= 0:
            return np.inf, np.zeros_like(weights)
        leverage = np.sum(coordinates * np.linalg.solve(moment, coordinates), axis=0)
        return -logdet, -leverage

    weights = np.full(shifts.shape[1], 1.0 / shifts.shape[1])
    if rank < shifts.shape[1]:
        fit = minimize(
            objective, weights, jac=True, method="SLSQP",
            bounds=[(0.0, 1.0)] * len(weights),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                         "jac": lambda w: np.ones_like(w)},
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not fit.success:
            raise RuntimeError(f"Model ellipsoid optimization failed: {fit.message}")
        weights = fit.x
    moment = rank * (coordinates * weights) @ coordinates.T
    distances = np.sum(coordinates * np.linalg.solve(moment, coordinates), axis=0)
    gap = max(0.0, float(distances.max()) - 1.0)
    if gap > 1e-6:
        raise RuntimeError(f"Model ellipsoid did not converge: containment gap {gap:g}")
    inflation = max(1.0, float(distances.max())) * (1.0 + 8 * np.finfo(float).eps)
    covariance = transform @ (inflation * moment) @ transform.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "rank": rank, "weights": weights.tolist(),
        "mahalanobis_squared": (distances / inflation).tolist(),
        "span_relative_residual": float(residual), "rank_rtol": rank_rtol,
        "optimality_gap": gap, "containment_inflation": inflation,
    }


def two_group_model_covariance(nominal, varied, normalization_weights):
    """Build group matrices from normalized densities, preserving all templates.

    normalization_weights has one row per pT slice: bin widths inside its
    normalization window and zero elsewhere. Already normalized template
    differences must satisfy these constraints; a material mismatch fails.
    Tiny floating-point leakage is removed before fitting the ellipsoids.
    """
    nominal = np.asarray(nominal, dtype=float)
    normalization_weights = np.asarray(normalization_weights, dtype=float)
    if nominal.ndim != 1 or not np.isfinite(nominal).all():
        raise ValueError("Expected a finite nominal density vector")
    if (normalization_weights.ndim != 2
            or normalization_weights.shape[1] != nominal.size
            or not np.isfinite(normalization_weights).all()):
        raise ValueError("Normalization weights do not match the nominal bins")
    matrices, diagnostics = {}, {}
    for group, sources in MODEL_GROUPS.items():
        shifts = np.column_stack([np.asarray(varied[name], dtype=float) - nominal for name in sources])
        if shifts.shape[0] != nominal.size or not np.isfinite(shifts).all():
            raise ValueError(f"Invalid normalized templates for {group}")
        leakage = normalization_weights @ shifts
        if np.max(np.abs(leakage), initial=0.0) > 1e-10:
            raise ValueError(f"Model templates for {group} violate per-pT normalization")
        # Rows have disjoint normalization windows. Correct within each window
        # only at roundoff level, without masking sparse or negative bins.
        for weights, leak in zip(normalization_weights, leakage):
            inside = weights != 0
            integral = weights @ nominal
            if integral == 0:
                raise ValueError("Empty model normalization window")
            shifts[inside] -= np.outer(nominal[inside] / integral, leak)
        matrices[group], diagnostics[group] = enclosing_template_covariance(shifts)
        diagnostics[group]["sources"] = list(sources)
        diagnostics[group]["max_input_normalization_leakage"] = float(np.max(np.abs(leakage), initial=0.0))
    return matrices, diagnostics
