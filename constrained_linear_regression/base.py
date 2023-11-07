"""Base of Constrained Machine Learning
"""

# Authors: Huijo Kim <huijo@hexafarms.com>
# License: BSD 3 clause
import abc
import numpy as np

try:
    from sklearn.linear_model._base import LinearModel, _preprocess_data
except ImportError:
    from sklearn.linear_model.base import LinearModel, _preprocess_data
from sklearn.neural_network import MLPRegressor
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
from sklearn.model_selection import train_test_split


class BaseConstrainedLinearRegression(LinearModel, RegressorMixin):
    def __init__(
        self,
        fit_intercept=True,
        normalize=False,
        copy_X=True,
        nonnegative=False,
        ridge=0,
        lasso=0,
        tol=1e-15,
        learning_rate=1.0,
        max_iter=10000,
        valid_ratio=0,
        training_losses=[],
        validation_losses=[],
    ):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.ridge = ridge
        self.lasso = lasso
        self.tol = tol
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.valid_ratio = valid_ratio
        self.training_losses = training_losses
        self.validation_losses = validation_losses

    def preprocess(self, X, y):
        X, y = check_X_y(
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            y_numeric=True,
            multi_output=False,
        )
        return _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
        )

    def _train_test_split(self, X, y):
        return train_test_split(X, y, test_size=self.valid_ratio, random_state=42)

    def _verify_coef(self, feature_count, coef, value, idx=0):
        if coef is not None:
            coef_ = coef
            assert (
                coef_.shape[-1] == feature_count
            ), "Incorrect shape for coef_, the second dimension must match feature_count"
        else:
            coef_ = np.ones((idx + 1, feature_count)) * value
        return coef_

    def _verify_initial_beta(self, feature_count, initial_beta):
        if initial_beta is not None:
            beta = initial_beta
            assert beta.shape == (feature_count,), "Incorrect shape for initial_beta"
        else:
            beta = np.zeros(feature_count).astype(float)
        return beta

    def _set_coef(self, beta):
        self.coef_ = beta

    def _save_mae(self, X, beta, y, loss_scale, X_valid, y_valid):
        self.training_losses.append(
            np.sum(np.abs(np.dot(X, beta) - y)) / loss_scale / (1 - self.valid_ratio)
        )
        if self.valid_ratio > 0:
            self.validation_losses.append(
                np.sum(np.abs(np.dot(X_valid, beta) - y_valid))
                / loss_scale
                / self.valid_ratio
            )

    @abc.abstractmethod
    def fit(self, X, y):
        pass


class BaseConstrainedMultilayerPerceptron(MLPRegressor, RegressorMixin):
    def __init__(
        self,
        hidden_layer_sizes=(10,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=False,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        assert shuffle is False, "shuffle should be False in the contrained ML."

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    def _verify_coef(self, feature_count, coef, value, idx=0):
        if coef is not None:
            coef_ = coef
            assert (
                coef_.shape[-1] == feature_count
            ), "Incorrect shape for coef_, the second dimension must match feature_count"
        else:
            coef_ = np.ones((idx + 1, feature_count)) * value
        return coef_
