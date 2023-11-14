import numpy as np
from sklearn.datasets import load_linnerud
from constrainedML.constrained_linear_regression import (
    ConstrainedLinearRegression,
)
from constrainedML.multi_constrained_linear_regression import (
    MultiConstrainedLinearRegression,
)
from constrainedML.constrained_multi_layer_perceptron import (
    ConstrainedMultilayerPerceptron,
)
from constrainedML.multi_constrained_multi_layer_perceptron import (
    MultiConstrainedMultilayerPerceptron,
)
from constrainedML.selective_drop_linear_regression import (
    SelectiveDropLinearRegression,
)
from constrainedML.selective_drop_multi_layer_perceptron import (
    SelectiveDropMultilayerPerceptron,
)
from constrainedML.multi_selective_drop_linear_regression import (
    MultiSelectiveDropLinearRegression,
)
from constrainedML.multi_selective_drop_multi_layer_perceptron import (
    MultiSelectiveDropMultilayerPerceptron,
)
from constrainedML.selective_drop_positive_linear_regression import (
    SelectiveDropPositiveLinearRegression,
)
from constrainedML.multi_selective_drop_positive_linear_regression import (
    MultiSelectiveDropPositiveLinearRegression,
)


def test_multi_selective_drop_lr():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = SelectiveDropPositiveLinearRegression()
    model = MultiSelectiveDropPositiveLinearRegression()

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        assert np.allclose(base_model.coef_, model.coef_), f"fails at {idx}th horizon."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_selective_drop_mlp_lbfgs(solver="lbfgs"):
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    random_state = 7
    hidden_layer_sizes = (3,)
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = SelectiveDropMultilayerPerceptron(
        solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )
    model = MultiSelectiveDropMultilayerPerceptron(
        solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        for layer_idx in range(model.n_layers_ - 1):
            assert np.allclose(
                base_model.coefs_[layer_idx], model.coefs_[layer_idx]
            ), f"fails at {idx}th horizon in {layer_idx}th layer."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_selective_drop_mlp_sgd(solver="sgd"):
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    random_state = 7
    hidden_layer_sizes = (3,)
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = SelectiveDropMultilayerPerceptron(
        solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )
    model = MultiSelectiveDropMultilayerPerceptron(
        solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        for layer_idx in range(model.n_layers_ - 1):
            assert np.allclose(
                base_model.coefs_[layer_idx], model.coefs_[layer_idx]
            ), f"fails at {idx}th horizon in {layer_idx}th layer."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_selective_drop_lr():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = SelectiveDropLinearRegression()
    model = MultiSelectiveDropLinearRegression()

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        assert np.allclose(base_model.coef_, model.coef_), f"fails at {idx}th horizon."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_constraint_mlp_lbfgs(solver="lbfgs"):
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    random_state = 7
    hidden_layer_sizes = (4,)
    horizon = 4
    min_coef = np.ones((horizon, 3)) * 0
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = 0
    min_coef[1, 1] = 1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 1

    max_coef[0, 0] = 0
    max_coef[1, 1] = 7
    max_coef[2, 1] = 5
    max_coef[3, 2] = 4

    base_model = ConstrainedMultilayerPerceptron(
        solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )
    model = MultiConstrainedMultilayerPerceptron(
        solver=solver, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state
    )

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        for layer_idx in range(model.n_layers_ - 1):
            assert np.allclose(
                base_model.coefs_[layer_idx], model.coefs_[layer_idx]
            ), f"fails at {idx}th horizon in {layer_idx}th layer."
        assert np.all(model.coefs_[0] >= 0), "clipping did not work well"
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_constraint_mlp_sgd(solver="sgd"):
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    random_state = 7
    hidden_layer_sizes = (3,)
    horizon = 4
    max_iter = 1000
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = ConstrainedMultilayerPerceptron(
        max_iter=max_iter,
        solver=solver,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
    )
    model = MultiConstrainedMultilayerPerceptron(
        max_iter=max_iter,
        solver=solver,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
    )

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        for layer_idx in range(model.n_layers_ - 1):
            assert np.allclose(
                base_model.coefs_[layer_idx], model.coefs_[layer_idx]
            ), f"fails at {idx}th horizon in {layer_idx}th layer."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_constraint():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    base_model = ConstrainedLinearRegression(nonnegative=False)
    model = MultiConstrainedLinearRegression(nonnegative=False)

    for idx in range(horizon):
        base_model.fit(X, y, min_coef=min_coef[idx], max_coef=max_coef[idx])
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
        assert np.allclose(base_model.coef_, model.coef_), f"fails at {idx}th horizon."
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


def test_multi_penalty_constraint():
    X, Y = load_linnerud(return_X_y=True)
    y = Y[:, 0]
    horizon = 4
    min_coef = np.ones((horizon, 3)) * -1
    max_coef = np.ones((horizon, 3)) * 2

    min_coef[0, 0] = -3
    min_coef[1, 1] = -1
    min_coef[2, 1] = 0
    min_coef[3, 2] = 0

    max_coef[0, 0] = -3
    max_coef[1, 1] = 1
    max_coef[2, 1] = 0
    max_coef[3, 2] = 0

    model = MultiConstrainedLinearRegression(nonnegative=False, penalty_rate=0.1)

    for idx in range(horizon):
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)
    model.reset()
    assert model.global_horizon_count == 0  # Check reset function


if __name__ == "__main__":
    test_multi_constraint_mlp_lbfgs()
