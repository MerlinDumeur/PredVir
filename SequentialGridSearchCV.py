from sklearn.model_selection import BaseSearchCV, ParameterGrid


class SequentialGridSearchCV(BaseSearchCV):

    def __init__(self, estimator, seq_param_grid, scoring=None, fit_params=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score="warn"):

        super().__init__(estimator=estimator, scoring=scoring, fit_params=fit_params,
                         n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)

        self.seq_param_grid = seq_param_grid

    def _run_search(self,evaluate_candidates):

        for param_grid in self.seq_param_grid:

            evaluate_candidates(ParameterGrid(param_grid))
