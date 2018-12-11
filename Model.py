import json
import CV
import zlib


def process_cv(params):

    ser_cv = CV.CV_FILE.serialize_CV(params['cv'])
    for k,v in ser_cv.items():

        params[k + '_cv'] = v

    del params['cv']

    return params


def process_model(params):

    ser_model = serialize_model(params['estimator'])

    for k,v in ser_model.items():

        params[k + '_model'] = v

    for k,v in params['param_grid'].items():

        # v.flags.writeable = False
        try:

            params['param_grid'][k] = zlib.adler32(v.tobytes())

        except AttributeError:

            params['param_grid'][k] = v

        # v.flags.writeable = True

    if params['scoring'].__class__.__name__ == '_ProbaScorer':

        params['scoring'] = params['scoring']._score_func.__name__

    else:

        raise(ValueError('Scoring method not supported'))
    
    del params['estimator']

    return params


def process_kernel(params,kernel):

    params.update(kernel.get_params())
    return params


def hash_model(params):

    return zlib.adler32(bytes(json.dumps(params, sort_keys=True),'utf-8'))


def serialize_model(model):

        return model.get_params()


class Model:

    def __init__(self,model,name):

        self.model = model
        self.name = name

    def __getattr__(self,*args,**kwargs):

        return self.model.__getattribute__(*args,**kwargs)

    def get_modelhash(self):

        view = self.model.get_params()
        params = {k:v for k,v in view.items()}
        params = process_model(params)
        params = process_cv(params)

        return hash_model(params)


class ModelKernel(Model):

    def __init__(self,model,name,kernel):

        Model.__init__(self,model,name)
        self.kernel = kernel

    def get_modelhash(self):

        params = self.model.get_params()
        params = process_kernel(params,self.kernel)
        params = process_model(params)
        params = process_cv(params)

        return hash_model(params)

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
