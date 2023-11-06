import numpy as np

from constrained_linear_regression.constrained_multi_layer_perceptron import (
    ConstrainedMultilayerPerceptron,
)


class MultiConstrainedMultilayerPerceptron(ConstrainedMultilayerPerceptron):
    global_horizon_count = 0

    def fit(self, X, y, min_coef=None, max_coef=None):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        feature_count = X.shape[-1]
        self.min_coef_ = self._verify_coef(
            feature_count,
            min_coef,
            -np.inf,
            MultiConstrainedMultilayerPerceptron.global_horizon_count,
        )
        self.max_coef_ = self._verify_coef(
            feature_count,
            max_coef,
            np.inf,
            MultiConstrainedMultilayerPerceptron.global_horizon_count,
        )

        return self._constrained_fit(X, y, incremental=False)

    def _update_coef_using_constrain(self, coef_grads):
        min_coefs = self.min_coef_[MultiConstrainedMultilayerPerceptron.global_horizon_count]
        max_coefs = self.max_coef_[MultiConstrainedMultilayerPerceptron.global_horizon_count]
        for coef_idx, (min_coef, max_coef) in enumerate(
            zip(min_coefs, max_coefs)
        ):
            # clipping is applied only to the first node.
            coef_grads[0][coef_idx] = np.clip(
                coef_grads[0][coef_idx],
                min_coef,
                max_coef,
            )

    def reset(self):
        MultiConstrainedMultilayerPerceptron.global_horizon_count = 0
        return f"horizon_count: {MultiConstrainedMultilayerPerceptron.global_horizon_count}"

    def _after_training(self):
        MultiConstrainedMultilayerPerceptron.global_horizon_count += 1