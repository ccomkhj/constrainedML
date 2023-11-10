import numpy as np
from .selective_drop_linear_regression import SelectiveDropLinearRegression


class MultiSelectiveDropLinearRegression(SelectiveDropLinearRegression):
    """
    This class implementation extends the SelectiveDropLinearRegression class to handle multiple constraints.

    Attributes
    ----------
    global_horizon_count : int
        Static/class-level attribute used to keep track of the number of constraints across multiple levels/horizons.

    Methods
    --------
    fit(self, X, y, min_coef=None, max_coef=None, initial_beta=None)
        Fit MultiSelectiveDropLinearRegression model to the given data.

    Parameters
    ----------
    X : array-like
        The input samples.

    y : array-like
        The target values.

    min_coef : array-like, default=None
        The minimum constraint for the coefficient. It's optional and default value is None

    max_coef : array-like
        The maximum constraint for the coefficient. It's optional and default value is None

    initial_beta : array-like
        Initial coefficients for starting the optimization process. It's optional and default value is None

    fit_intercept : bool, default=True
        Option to fit intercept.

    normalize : bool, default=False
        Option to normalize the data.

    copy_X : bool, default=True
        Option to copy X.

    nonnegative : bool, default=False
        Option to make sure coefficients are non-negative.

    ridge : int, default=0
        Regularization factor for ridge regression.

    lasso : int, default=0
        Regularization factor for lasso regression.

    tol : float, default=1e-15
        Tolerance for the stopping criterion.

    learning_rate : float, default=1.0
        Learning rate for the update rule.

    max_iter : int, default=10000
        Maximum number of iterations for the algorithm.

    penalty_rate : float, default=0
        Penalty rate applied to the coefficients.

    Return
    -------
    self : object
        Returns the instance itself.
    """

    global_horizon_count = 0  # Class attribute shared by all instances

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
        penalty_rate=0,
        valid_ratio=0,
        training_losses=[],
        validation_losses=[],
        n_th_model=0,
        min_coef_=None,
        max_coef_=None,
    ):
        super().__init__(
            fit_intercept,
            normalize,
            copy_X,
            nonnegative,
            ridge,
            lasso,
            tol,
            learning_rate,
            max_iter,
            valid_ratio,
            training_losses,
            validation_losses,
        )
        self.penalty_rate = penalty_rate
        self.n_th_model = n_th_model
        self.max_coef_ = min_coef_
        self.min_coef_ = max_coef_

    def fit(
        self,
        X,
        y,
        min_coef=None,
        max_coef=None,
        initial_beta=None,
    ):
        X, y, X_offset, y_offset, X_scale = self.preprocess(X, y)

        original_feature_count = X.shape[-1]
        self.min_coef_ = self._verify_coef(
            original_feature_count,
            min_coef,
            -np.inf,
            MultiSelectiveDropLinearRegression.global_horizon_count,
        )
        self.max_coef_ = self._verify_coef(
            original_feature_count,
            max_coef,
            np.inf,
            MultiSelectiveDropLinearRegression.global_horizon_count,
        )

        selected_features = self._find_non_zero_indices(
            self.min_coef_[MultiSelectiveDropLinearRegression.global_horizon_count],
            self.max_coef_[MultiSelectiveDropLinearRegression.global_horizon_count],
        )
        feature_count = len(selected_features)

        # Keep only selected features
        X = X[:, selected_features]
        X_offset = X_offset[selected_features]
        X_scale = X_scale[selected_features]

        if self.valid_ratio > 0:
            # Split the data into training and validation sets
            X, X_valid, y, y_valid = self._train_test_split(X, y)
        else:
            X_valid, y_valid = None, None

        beta = self._verify_initial_beta(feature_count, initial_beta)

        if self.nonnegative:
            self.min_coef_ = np.clip(self.min_coef_, 0, None)

        prev_beta = beta + 1
        hessian = self._calculate_hessian(X)
        loss_scale = len(y)

        # Custom fit implementation starts from here
        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print("THE MODEL DID NOT CONVERGE")
                break

            step += 1
            prev_beta = beta.copy()

            self._save_mae(X, beta, y, loss_scale, X_valid, y_valid)

            for i, _ in enumerate(beta):
                grad = self._calculate_gradient(X, beta, y)
                if self.penalty_rate:
                    progress = step / self.max_iter
                    grad += (
                        progress
                        * self.penalty_rate
                        * self._calc_distance_out_of_bounds(beta, i, selected_features)
                    )

                beta[i] = self._update_beta(
                    beta,
                    i,
                    grad,
                    hessian,
                    loss_scale,
                    self.min_coef_[
                        MultiSelectiveDropLinearRegression.global_horizon_count
                    ][selected_features],
                    self.max_coef_[
                        MultiSelectiveDropLinearRegression.global_horizon_count
                    ][selected_features],
                )

        self._set_coef(beta)
        self._set_intercept(X_offset, y_offset, X_scale)

        # Register for prediction and Update horizon_count for the next model
        self.n_th_model = MultiSelectiveDropLinearRegression.global_horizon_count
        MultiSelectiveDropLinearRegression.global_horizon_count += 1

        return self

    def _calc_distance_out_of_bounds(self, beta, i, selected_features):
        min_bound = self.min_coef_[
            MultiSelectiveDropLinearRegression.global_horizon_count
        ][selected_features][i]
        max_bound = self.max_coef_[
            MultiSelectiveDropLinearRegression.global_horizon_count
        ][selected_features][i]
        if beta[i] < min_bound:
            return beta[i] - min_bound
        elif beta[i] > max_bound:
            return beta[i] - max_bound
        else:
            return 0

    def reset(self):
        MultiSelectiveDropLinearRegression.global_horizon_count = 0
        return (
            f"horizon_count: {MultiSelectiveDropLinearRegression.global_horizon_count}"
        )
