import GeneSelection
import Constants

import pandas as pd
import numpy as np


class ModelTester:

    def __init__(self,model):

        # Model needs to implement hyperparameter fitting (similar to GridSearchCV)

        self.model = model

    def fit_procedure(self,ds,cv_primary,strata,geneSelector,X,Y):

        values_params = {k:[] for k in self.model.param_grid}

        for i,(Xtrain,Ytrain,Xtest,Ytest) in enumerate(ds.CV_split(cv_primary,strata)):

            if issubclass(geneSelector.__class__,GeneSelection.GeneSelector):

                index_genes = geneSelector.select_genes(X,Y)

            else:

                index_genes = geneSelector.select_genes(i)

            Xtrain = Xtrain.loc[:,index_genes]
            Xtest = Xtest.loc[:,index_genes]

            self.model.fit(Xtrain,Ytrain)

            bp = self.model.best_params

            for k,v in values_params.items():

                v.append(bp[k])

        return values_params

    def test_score(self,ds,cv_primary,geneSelector,metrics={},strata=None,save=None,adaptative=False):

        X, Y = ds.X, ds.Y

        MI = pd.MultiIndex.from_arrays([['general'] * 2,['fs_size','Class']])

        arrays_m = [['Test'] * (len(metrics) + 1),['score',*metrics]]
        MI2 = pd.MultiIndex.from_arrays(arrays_m)

        param_grid = self.model.param_grid
        arrays_p = [['Validation'] * (len(param_grid) + 1),['score',*param_grid]]
        MI3 = pd.MultiIndex.from_arrays(arrays_p)

        MI_final = MI.copy()
        MI_final = MI_final.append(MI2)
        MI_final = MI_final.append(MI3)

        df_output = pd.DataFrame(index=np.arange(cv_primary.get_n_splits()),columns=MI_final)

        if adaptative:

            i = 0
            cont = True

            while cont and i < self.model.n_iter_search:

                values_params = self.fit_procedure(ds,cv_primary,strata,geneSelector,X,Y)
                cont = self.model.update_intervals(values_params)

        for i,(Xtrain,Ytrain,Xtest,Ytest) in enumerate(ds.CV_split(cv_primary,strata)):

            if issubclass(geneSelector.__class__,GeneSelection.GeneSelector):

                index_genes = geneSelector.select_genes(X,Y)

            else:

                index_genes = geneSelector.select_genes(i)

            Xtrain = Xtrain.loc[:,index_genes]
            Xtest = Xtest.loc[:,index_genes]

            self.model.fit(Xtrain,Ytrain)

            best_params = self.model.best_params_
            score = self.model.score(Xtest,Ytest)

            df_output.loc[i,'general'] = Xtrain.shape[1]
            df_output.loc[i,'Class'] = self.model.best_estimator_.__class__.__name__

            if len(metrics) == 0:
                df_output.loc[i,'Test'] = score
            else:
                # print(Ytest.shape)
                # print(self.model.predict_proba(Xtest).shape)
                # print(metrics['accuracy'](Ytest,self.model.predict(Xtest)))
                df_output.loc[i,'Test'] = [score] + [metrics[m](Ytest,self.model.predict_proba(Xtest)[:,1]) if metrics[m].__name__ not in Constants.no_proba_list else metrics[m](Ytest,self.model.predict(Xtest)) for m in arrays_m[1][1:]]

            if len(best_params) == 0:
                df_output.loc[i,'Validation'] = self.model.best_score_
            else:
                df_output.loc[i,'Validation'] = [self.model.best_score_] + [best_params[k] for k in arrays_p[1][1:]]

        if save is not None:

            df_output.to_pickle(save)

        return df_output
