import json


def get_modelhash(model):

    return hash(json.dumps(model.get_params(), sort_keys=True))

# class ModelTester:

#     def __init__(self,model):

#         # Model needs to implement hyperparameter fitting (similar to GridSearchCV)

#         self.model = model

#     def test_score(self,X,Y,cv_primary,metrics={},strata=None,save=None):

#         MI = pd.MultiIndex.from_arrays([['general'],['fs_size']])
        
#         arrays_m = [['Test'] * (len(metrics) + 1),['score',*metrics.keys()]]
#         MI2 = pd.MultiIndex.from_arrays(arrays_m)

#         param_grid = param[self.predictorCV.param_grid_name]
#         arrays_p = [['Validation'] * (len(param_grid) + 1),['score',*param_grid]]
#         MI3 = pd.MultiIndex.from_arrays(arrays_p)
        
#         MI_final = MI.copy()
#         MI_final = MI_final.append(MI2)
#         MI_final = MI_final.append(MI3)

#         df_output = pd.DataFrame(index=np.arange(dataSet.cv_primary.get_n_splits()),columns=MI_final)

#         for train_index,test_index in CV.split(X,strata):

#             Xtrain = X.loc[train_index]
#             Ytrain = Y.loc[train_index]

#             Xtest = X.loc[test_index]
#             Ytest = Y.loc[test_index]

#             self.model.fit(Xtrain,Ytrain)

#             best_params = self.model.best_params()
#             score = self.predictorCV.score(Xtest,Ytest)

#             df_output.loc[i,'general'] = Xval.shape[1]

#             if len(metrics) == 0:
#                 df_output.loc[i,'Test'] = score
#             else:
#                 df_output.loc[i,'Test'] = [score] + [metrics[m](Ytest,self.predictorCV.predict_proba(Xtest),labels=[0,1]) for m in arrays_m[1][1:]]
            
#             if len(best_params) == 0:
#                 df_output.loc[i,'Validation'] = self.predictorCV.best_score()
#             else:
#                 df_output.loc[i,'Validation'] = [self.predictorCV.best_score()] + [best_params[k] for k in arrays_p[1][1:]]

#         if save is not None:

#             df_output.to_pickle(save)

#         return df_output
