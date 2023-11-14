"""
Huijo Kim (huijo@hexafarms.com)
"""
from itertools import chain
import numpy as np

import scipy.optimize

from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils import check_random_state

from constrainedML.selective_drop_multi_layer_perceptron import (
    SelectiveDropMultilayerPerceptron,
    _pack,
)


class MultiSelectiveDropMultilayerPerceptron(SelectiveDropMultilayerPerceptron):
    global_horizon_count = 0

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
        training_losses=[],
        n_th_model=0,
        min_coef_=None,
        max_coef_=None,
        selected_features=[],
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
            training_losses=training_losses,
            n_th_model=n_th_model,
            max_coef_=min_coef_,
            min_coef_=max_coef_,
            selected_features=selected_features,
        )

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
        original_feature_count = X.shape[-1]
        self.min_coef_ = self._verify_coef(
            original_feature_count,
            min_coef,
            -np.inf,
            MultiSelectiveDropMultilayerPerceptron.global_horizon_count,
        )
        self.max_coef_ = self._verify_coef(
            original_feature_count,
            max_coef,
            np.inf,
            MultiSelectiveDropMultilayerPerceptron.global_horizon_count,
        )

        self.selected_features = self._find_non_zero_indices(
            self.min_coef_[MultiSelectiveDropMultilayerPerceptron.global_horizon_count],
            self.max_coef_[MultiSelectiveDropMultilayerPerceptron.global_horizon_count],
        )
        # Keep only selected features
        X = X[:, self.selected_features]

        return self._constrained_fit(X, y, incremental=False)

    def _update_coef_using_constrain(self, coef_grads):
        min_coefs = self.min_coef_[
            MultiSelectiveDropMultilayerPerceptron.global_horizon_count
        ][self.selected_features]
        max_coefs = self.max_coef_[
            MultiSelectiveDropMultilayerPerceptron.global_horizon_count
        ][self.selected_features]

        for coef_idx, (min_coef, max_coef) in enumerate(zip(min_coefs, max_coefs)):
            # clipping is applied only to the first node.
            coef_grads[0][coef_idx] = np.clip(
                coef_grads[0][coef_idx],
                min_coef,
                max_coef,
            )

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
            raise KeyError(f"unknown solver: {self.solver}")

        # validate parameter weights
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self

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
                    MultiSelectiveDropMultilayerPerceptron.global_horizon_count
                ][self.selected_features],
                self.max_coef_[
                    MultiSelectiveDropMultilayerPerceptron.global_horizon_count
                ][self.selected_features],
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
        MultiSelectiveDropMultilayerPerceptron.global_horizon_count = 0
        return f"horizon_count: {MultiSelectiveDropMultilayerPerceptron.global_horizon_count}"

    def _after_training(self):
        self.n_th_model = MultiSelectiveDropMultilayerPerceptron.global_horizon_count
        MultiSelectiveDropMultilayerPerceptron.global_horizon_count += 1

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
        grad = _pack(coef_grads, intercept_grads)
        self._save_mse(loss)
        return loss, grad
