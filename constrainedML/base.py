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
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_X_y
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.model_selection import train_test_split
from scipy import sparse


class BaseSelectiveDropPositiveLinearRegression(RegressorMixin, LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

        .. deprecated:: 1.0
           `normalize` was deprecated in version 1.0 and will be
           removed in 1.2.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        `n_targets > 1` and secondly `X` is sparse or if `positive` is set
        to `True`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive. This
        option is only supported for dense arrays.

        .. versionadded:: 0.24

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    rank_ : int
        Rank of matrix `X`. Only available when `X` is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of `X`. Only available when `X` is dense.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression addresses some of the
        problems of Ordinary Least Squares by imposing a penalty on the
        size of the coefficients with l2 regularization.
    Lasso : The Lasso is a linear model that estimates
        sparse coefficients with l1 regularization.
    ElasticNet : Elastic-Net is a linear regression
        model trained with both l1 and l2 -norm regularization of the
        coefficients.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) or Non Negative Least Squares
    (scipy.optimize.nnls) wrapped as a predictor object.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.score(X, y)
    1.0
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(
        self,
        *,
        fit_intercept=False,
        copy_X=True,
        n_jobs=None,
        positive=True,
    ):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def _preprocess(self, X, y, sample_weight=None):
        """
        preprocess dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.

        Returns
        -------
        self : object
            Fitted Estimator.
        """

        n_jobs_ = self.n_jobs

        accept_sparse = False if self.positive else ["csr", "csc", "coo"]

        X, y = self._validate_data(
            X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
        )

        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype, only_non_negative=True
        )

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            normalize=None,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        # Sample weight can be implemented via a simple rescaling.
        X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)

        return X, y, X_offset, y_offset, X_scale

    def _verify_coef(self, feature_count, coef, value, idx=0):
        if coef is not None:
            coef_ = coef
            assert (
                coef_.shape[-1] == feature_count
            ), "Incorrect shape for coef_, the second dimension must match feature_count"
        else:
            coef_ = np.ones((idx + 1, feature_count)) * value
        return coef_


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
        training_losses=[],
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
        self.training_losses = training_losses
        assert shuffle is False, "shuffle should be False in the contrained ML."

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    def _save_mse(self, loss):
        self.training_losses.append(loss)

    def _verify_coef(self, feature_count, coef, value, idx=0):
        if coef is not None:
            coef_ = coef
            assert (
                coef_.shape[-1] == feature_count
            ), "Incorrect shape for coef_, the second dimension must match feature_count"
        else:
            coef_ = np.ones((idx + 1, feature_count)) * value
        return coef_


def _rescale_data(X, y, sample_weight):
    """Rescale data sample-wise by square root of sample_weight.

    For many linear models, this enables easy support for sample_weight because

        (y - X w)' S (y - X w)

    with S = diag(sample_weight) becomes

        ||y_rescaled - X_rescaled w||_2^2

    when setting

        y_rescaled = sqrt(S) y
        X_rescaled = sqrt(S) X

    Returns
    -------
    X_rescaled : {array-like, sparse matrix}

    y_rescaled : {array-like, sparse matrix}
    """
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y, sample_weight_sqrt
