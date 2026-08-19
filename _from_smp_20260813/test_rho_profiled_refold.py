import numpy as np

from scripts.studies.unfold.rho_profiled_refold import (
    Nuisance,
    solve_profiled,
    solve_unconstrained_gls,
)


def test_profiled_nuisance_reduces_an_out_of_model_residual():
    response = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    truth = np.array([10.0, 20.0])
    nuisance_direction = np.array([1.0, -1.0, 2.0])
    covariance = np.eye(3) * 0.01
    data = response @ truth - 1.5 * nuisance_direction

    baseline = solve_unconstrained_gls(data, covariance, response)
    profiled = solve_profiled(
        data,
        covariance,
        response,
        [
            Nuisance(
                name="test",
                response_derivative=np.zeros_like(response),
                data_derivative=nuisance_direction,
            )
        ],
        baseline["truth"],
    )

    assert profiled["success"]
    assert profiled["chi2_total"] < baseline["chi2_total"]
    assert np.isclose(profiled["nuisance_pulls"]["test"], 1.5, rtol=0.02)
    assert np.allclose(profiled["truth"], truth, rtol=0.01)
