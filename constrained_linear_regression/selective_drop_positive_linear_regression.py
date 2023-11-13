from constrained_linear_regression.base import BaseSelectiveDropPositiveLinearRegression

from scipy import optimize
from joblib import Parallel
from sklearn.utils.fixes import delayed
import numpy as np


class SelectiveDropPositiveLinearRegression(BaseSelectiveDropPositiveLinearRegression):
    """ """

    def fit(self, X, y, min_coef=None, max_coef=None):
        X, y, X_offset, y_offset, X_scale = self._preprocess(X, y)
        feature_count = X.shape[-1]
        self.min_coef_ = self._verify_coef(feature_count, min_coef, -np.inf).flatten()
        self.max_coef_ = self._verify_coef(feature_count, max_coef, np.inf).flatten()

        selected_features = self._find_non_zero_indices(
            self.min_coef_,
            self.max_coef_,
        )
        feature_count = len(selected_features)

        # Keep only selected features
        X = X[:, selected_features]
        X_offset = X_offset[selected_features]
        X_scale = X_scale[selected_features]

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
        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """

        selected_features = self._find_non_zero_indices(
            self.min_coef_[self.n_th_model],
            self.max_coef_[self.n_th_model],
        )
        X = X[:, selected_features]

        return self._decision_function(X)

    def _find_non_zero_indices(self, max_coef, min_coef):
        if len(max_coef) != len(min_coef):
            raise ValueError("The input arrays must have the same length")

        # Find the indices where both array elements are not 0
        non_zero_indices = np.where((max_coef != 0) | (min_coef != 0))[0]
        return non_zero_indices
