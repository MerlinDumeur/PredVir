import numpy as np

from sklearn.model_selection import BaseSearchCV, ParameterGrid


f_gen_dict = {
    'linear':np.linspace,
    'logarithmic':np.logspace
}

f_inverse_dict = {

    'linspace': lambda x:x,
    'logspace': np.log

}


@dataclass
class ParameterInterval:

    name: str
    init_interval: List[int]
    max_interval: List[int]
    n_val: int
    f_gen

# class ParameterInterval:

#     def __init__(self,name,init_interval,max_interval,n_val,f_gen):

#         self.name = name
#         self.init_interval = init_interval
#         self.max_interval = max_interval
#         self.n_val = n_val
#         self.f_gen = f_gens


class AdaptativeParameterGrid:

    interval: Dict[str,List[int]] = {}
    max_interval: Dict[str,List[int]] = {}
    n_val: Dict[str,int] = {}

    def __init__(self,PI_list,p_aug=0.3):

        for PI in PI_list:

            name = PI.name
            self.interval[name] = PI.init_interval
            self.max_interval[name] = PI.max_interval
            self.n_val[name] = PI.n_val
            self.f_gen[name] = f_gen_dict[PI.f_gen] if isinstance(PI.f_gen,str) else PI.f_gen
            self.f_inverse[name] = f_inverse_dict[self.f_gen[name].__name__]

        self.p_aug = p_aug

    def get_param_grid(self):

        self.param_grid = {k:f(self.interval[k][0],self.interval[k][1],self.n_val[k]) for k,f in self.f_gen.items()}
        return ParameterGrid(self.param_grid)

    def update_intervals(self,used_values):

        for name,values in used_values:

            min_val = min(values)
            max_val = max(values)

            fitting_min = False
            fitting_max = False

            if min_val > min(self.param_grid[name]):

                self.interval[name][0] = self.f_inverse[name](min_val)

            elif np.mean(values == min_val) > self.p_aug:

                self.interval[name][0] -= abs(np.mean(values == min_val) * (self.interval[name][0] - self.max_interval[name][0]))

            else:

                fitting_min = True

            if max_val < max(self.param_grid[name]):

                self.interval[name][1] = self.f_inverse[name](max_val)

            elif np.mean(values == max_val) > self.p_aug:

                self.interval[name][0] += abs(np.mean(values == max_val) * (self.max_interval[name][1] - self.interval[name][1]))

            else:

                fitting_max = True

            if self.interval[name][0] > self.interval[name][1]:

                self.interval[name][0],self.interval[name][1] = self.interval[name][1],self.interval[name][0]

            return (fitting_max and fitting_min)


class AdaptativeSearchCV(BaseSearchCV):

    def __init__(self, estimator, adapt_param_grid, scoring=None, fit_params=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score="warn",n_iter_search=5):

        super().__init__(estimator=estimator, scoring=scoring, fit_params=fit_params,
                         n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)

        self.adapt_param_grid = adapt_param_grid
        self.n_iter_search = n_iter_search

    def _run_search(self,evaluate_candidates):

        evaluate_candidates(self.adapt_param_grid.get_param_grid())