import numpy as np
from itertools import chain
import warnings

import scipy.optimize

from sklearn.utils.optimize import _check_optimize_result
from sklearn.base import is_classifier
from sklearn.utils import gen_batches, check_random_state
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from sklearn.model_selection import train_test_split
from .base import BaseConstrainedMultilayerPerceptron


class ConstrainedMultilayerPerceptron(BaseConstrainedMultilayerPerceptron):
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
        self.min_coef_ = self._verify_coef(feature_count, min_coef, -np.inf).flatten()
        self.max_coef_ = self._verify_coef(feature_count, max_coef, np.inf).flatten()

        return self._constrained_fit(X, y, incremental=False)

    def _constrained_fit(self, X, y, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X, y = self._validate_input(X, y, incremental, reset=first_pass)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, layer_units, X.dtype)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # Run the Stochastic optimization solver
        if self.solver in ["sgd", "adam"]:
            self._fit_constrained_stochastic(
                X,
                y,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )

        # Run the LBFGS solver
        elif self.solver == "lbfgs":
            self._fit_constrained_lbfgs(
                X, y, activations, deltas, coef_grads, intercept_grads, layer_units
            )

        else:
            raise f"unknown solver: {self.solver}"

        # validate parameter weights
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self

    def _fit_constrained_stochastic(
        self,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
    ):
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                stratify=stratify,
            )
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    # It doesn't shuffle because the sequence matters in the constrained ML.
                    X_batch = X[batch_slice]
                    y_batch = y[batch_slice]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch,
                        y_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                    )
                    accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)
                    self._update_coef_using_constrain(params)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than "
                            "tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f"
                            " for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter
                    )
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

        self._after_training()

    def _update_coef_using_constrain(self, coef_grads):
        for coef_idx, (min_coef, max_coef) in enumerate(
            zip(self.min_coef_, self.max_coef_)
        ):
            # clipping is applied only to the first node.
            coef_grads[0][coef_idx] = np.clip(
                coef_grads[0][coef_idx], min_coef, max_coef
            )

    def _after_training(self):
        # This is useful only for multi models.
        pass

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

        opt_res = scipy.optimize.minimize(
            self._loss_grad_constrained_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
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

    def _loss_grad_constrained_lbfgs(
        self, packed_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads
    ):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.

        Returned gradients are packed in a single vector so it can be used
        in lbfgs

        Parameters
        ----------
        packed_coef_inter : ndarray
            A vector comprising the flattened coefficients and intercepts.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads = self._backprop(
            X, y, activations, deltas, coef_grads, intercept_grads
        )
        self._update_coef_using_constrain(coef_grads)  # it throws error.
        grad = _pack(coef_grads, intercept_grads)
        self._save_mse(loss)
        return loss, grad


def _pack(coefs_, intercepts_):
    """Pack the parameters into a single vector."""
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])
