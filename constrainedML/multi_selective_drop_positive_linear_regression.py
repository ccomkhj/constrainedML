import numpy as np

from scipy import optimize
from joblib import Parallel
from sklearn.utils.fixes import delayed

from .selective_drop_positive_linear_regression import (
    SelectiveDropPositiveLinearRegression,
)


class MultiSelectiveDropPositiveLinearRegression(SelectiveDropPositiveLinearRegression):
    """ """

    global_horizon_count = 0  # Class attribute shared by all instances

    def __init__(
        self,
        *,
        fit_intercept=False,
        copy_X=True,
        n_jobs=None,
        positive=True,
        n_th_model=0,
        min_coef_=None,
        max_coef_=None,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self.n_th_model = n_th_model
        self.max_coef_ = min_coef_
        self.min_coef_ = max_coef_

    def fit(self, X, y, min_coef=None, max_coef=None):
        X, y, X_offset, y_offset, X_scale = self._preprocess(X, y)

        original_feature_count = X.shape[-1]
        self.min_coef_ = self._verify_coef(
            original_feature_count,
            min_coef,
            -np.inf,
            MultiSelectiveDropPositiveLinearRegression.global_horizon_count,
        )
        self.max_coef_ = self._verify_coef(
            original_feature_count,
            max_coef,
            np.inf,
            MultiSelectiveDropPositiveLinearRegression.global_horizon_count,
        )

        selected_features = self._find_non_zero_indices(
            self.min_coef_[
                MultiSelectiveDropPositiveLinearRegression.global_horizon_count
            ],
            self.max_coef_[
                MultiSelectiveDropPositiveLinearRegression.global_horizon_count
            ],
        )
        feature_count = len(selected_features)

        # Keep only selected features
        X = X[:, selected_features]
        X_offset = X_offset[selected_features]
        X_scale = X_scale[selected_features]
        self.n_features_in_ = feature_count

        if self.positive:
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else:
                # scipy.optimize.nnls cannot handle y with shape (M, K)
                outs = Parallel(n_jobs=self.n_jobs)(
                    delayed(optimize.nnls)(X, y[:, j]) for j in range(y.shape[1])
                )
                self.coef_ = np.vstack([out[0] for out in outs])

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)

        self.n_th_model = (
            MultiSelectiveDropPositiveLinearRegression.global_horizon_count
        )
        MultiSelectiveDropPositiveLinearRegression.global_horizon_count += 1

        return self

    def reset(self):
        MultiSelectiveDropPositiveLinearRegression.global_horizon_count = 0
        return f"horizon_count: {MultiSelectiveDropPositiveLinearRegression.global_horizon_count}"
