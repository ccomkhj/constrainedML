import numpy as np
from itertools import chain

from constrainedML.constrained_multi_layer_perceptron import (
    ConstrainedMultilayerPerceptron,
    _pack,
)
import scipy.optimize

from sklearn.utils.optimize import _check_optimize_result


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
        min_coefs = self.min_coef_[
            MultiConstrainedMultilayerPerceptron.global_horizon_count
        ]
        max_coefs = self.max_coef_[
            MultiConstrainedMultilayerPerceptron.global_horizon_count
        ]
        for coef_idx, (min_coef, max_coef) in enumerate(zip(min_coefs, max_coefs)):
            # clipping is applied only to the first node.
            coef_grads[0][coef_idx] = np.clip(
                coef_grads[0][coef_idx],
                min_coef,
                max_coef,
            )

    def _fit_constrained_lbfgs(
        self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units
    ):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_, self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        bounds = [
            [(min_val, max_val)] * self.hidden_layer_sizes[0]
            for min_val, max_val in zip(
                self.min_coef_[
                    MultiConstrainedMultilayerPerceptron.global_horizon_count
                ],
                self.max_coef_[
                    MultiConstrainedMultilayerPerceptron.global_horizon_count
                ],
            )
        ]
        bounds = list(chain(*bounds))
        effective_bounds_len = len(bounds)
        for _ in range(len(packed_coef_inter) - effective_bounds_len):
            bounds.append((-np.inf, np.inf))

        opt_res = scipy.optimize.minimize(
            self._loss_grad_constrained_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxfun": self.max_fun,
                "maxiter": self.max_iter,
                "iprint": iprint,
                "gtol": self.tol,
            },
            args=(X, y, activations, deltas, coef_grads, intercept_grads),
        )
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)
        self._after_training()

    def reset(self):
        MultiConstrainedMultilayerPerceptron.global_horizon_count = 0
        return f"horizon_count: {MultiConstrainedMultilayerPerceptron.global_horizon_count}"

    def _after_training(self):
        MultiConstrainedMultilayerPerceptron.global_horizon_count += 1
