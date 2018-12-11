import numbers
import warnings

import numpy as np
# from scipy import optimize, sparse
# from scipy.special import expit

# from .base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
# from .sag import sag_solver
# from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.svm.base import _fit_liblinear
# from sklearn.utils import check_array, check_consistent_length, compute_class_weight
# from sklearn.utils import check_random_state
# from sklearn.utils.extmath import (log_logistic, safe_sparse_dot, softmax,
#                               squared_norm)
from sklearn.utils.extmath import softmax
from sklearn.utils.extmath import row_norms
# from sklearn.utils.fixes import logsumexp
# from sklearn.utils.optimize import newton_cg
from sklearn.utils.validation import check_X_y
# from sklearn.exceptions import (ConvergenceWarning,ChangedBehaviorWarning)
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.externals.joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils.fixes import _joblib_parallel_args
# from sklearn.model_selection import check_cv
# from sklearn.externals import six
# from sklearn.metrics import get_scorer

from sklearn.linear_model.logistic import _check_solver,_check_multi_class
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels


class KernelLogisticRegression(LogisticRegression):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 kernel='linear',gamma=None,degree=3,coef0=1,kernel_params=None,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None):

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

        super().__init__(penalty, dual, tol, C,
                         fit_intercept, intercept_scaling, class_weight,
                         random_state, solver, max_iter,
                         multi_class, verbose, warm_start, n_jobs)

    def _get_kernel(self,X,Y=None):

        if callable(self.kernel):

            params = self.kernel_params or {}

        else:

            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}

        return pairwise_kernels(X,Y,metric=self.kernel,filter_params=True,**params)

    def fit(self,X,y,sample_weight=None):

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        solver = _check_solver(self.solver, self.penalty, self.dual)

        if solver in ['newton-cg']:
            _dtype = [np.float64, np.float32]
        else:
            _dtype = np.float64

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype, order="C",
                         accept_large_sparse=solver != 'liblinear')

        self.X_fit_ = X
        X_k = self._get_kernel(X)

        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X_k.shape

        multi_class = _check_multi_class(self.multi_class, solver,
                                               len(self.classes_))

        if solver == 'liblinear':
            if effective_n_jobs(self.n_jobs) != 1:
                warnings.warn("'n_jobs' > 1 does not have any effect when"
                              " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                              " = {}.".format(effective_n_jobs(self.n_jobs)))
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X_k, y, self.C, self.fit_intercept, self.intercept_scaling,
                self.class_weight, self.penalty, self.dual, self.verbose,
                self.max_iter, self.tol, self.random_state,
                sample_weight=sample_weight)
            self.n_iter_ = np.array([n_iter_])
            return self

        if solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X_k, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == 'multinomial':
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(super.logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if solver in ['sag', 'saga']:
            prefer = 'threads'
        else:
            prefer = 'processes'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer=prefer))(
            path_func(X_k, y, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, tol=self.tol,
                      verbose=self.verbose, solver=solver,
                      multi_class=multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      penalty=self.penalty,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if multi_class == 'multinomial':
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(n_classes, n_features +
                                            int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self

    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")

        X_k = self._get_kernel(X,self.X_fit_)

        ovr = (self.multi_class in ["ovr", "warn"] or
               (self.multi_class == 'auto' and (self.classes_.size <= 2 or
                                                self.solver == 'liblinear')))
        if ovr:
            return super(LogisticRegression, self)._predict_proba_lr(X_k)
        else:
            decision = self.decision_function(X_k)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)
