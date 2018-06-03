from sklearn.metrics import log_loss,make_scorer
from sklearn.model_selection import RepeatedKFold,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso,Ridge
import Traitement as proc
import Predicteurs as cl
import pandas as pd
import pickle
import os
import numpy as np

CV_dict = {
    'KFold':'KF',
    'RepeatedKFold':'RKF',
    'StratifiedKFold':'SKF',
    'RepeatedStratifiedKFold':'SRKF'
}


def score_predicteur(X,Y,classifieur,classifieur_frelevance,classifieur_fselect,weight_fs,grid_fs,params_grid,metric_fs=log_loss,cv_main=KFold(n_splits=3,shuffle=True),cv_grid=KFold(n_splits=5,shuffle=True),smoothing_fs=lambda m:proc.smooth(m,window_len=5,window='hanning'),**kwargs):

    scoring = kwargs.get('scoring',make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1]))
    metrics_test = kwargs.get('metrics',{'log_loss':log_loss})

    GdS = GridSearchCV(classifieur.predicteur,params_grid,scoring=scoring,n_jobs=-1,cv=cv_grid,return_train_score=False)

    msx_list = []

    Index_results = ['score','log_loss','fs_size'] + [k for k in params_grid]
    results = pd.DataFrame(columns=Index_results)

    for IndexValidation,IndexTest in cv_main.split(X,Y):

        Xval,Yval = X.iloc[IndexValidation],Y.iloc[IndexValidation]
        Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]

        I,msx = classifieur_fselect.feature_select_from_model_weights(Xval,Yval,classifieur_frelevance,{"metric_fs":metric_fs},weight_fs,grid_fs,smoothing_fs)

        msx_list.append(msx)

        GdS.fit(Xval[I],Yval)
        estimator = GdS.best_estimator_
        data = GdS.best_params_.copy()

        data['log_loss'] = log_loss(Ytest,estimator.predict_proba(Xtest[I]),labels=[0,1])
        data['score'] = estimator.score(Xtest[I],Ytest)
        data['fs_size'] = len(I)

        S = pd.Series(data=data,index=Index_results)

        results = results.append(S,ignore_index=True)
        
    return results,msx_list


def get_filename(cv_primary,id,nmois=None,std=True):

    predicteur_str = 'R' if nmois is None else f'C-{nmois}'
    
    filename = rf'{predicteur_str}.{"S" if std else ""}.{CV_dict[type(cv_primary).__name__]}.{cv_primary.get_n_splits()}-{id}.pkl'

    return filename


def get_foldername(type,base,classifieur_fr=None,classifieur_fs=None,classifieur_model=None,**kwargs):

    filename1 = base + rf'/cv-sets/'

    if type == 'CV_primary':

        return filename1

    filename2 = filename1 + rf'{classifieur_fr.name}-{classifieur_fs.name}/'

    if type == 'FS':

        return filename2

    elif type == 'FS+CV':

        return filename1,filename2

    filename3 = filename2 + rf'{classifieur_model.name}/'

    if type == 'model_scoring':

        return filename3

    elif type == 'all':

        return filename1,filename2,filename3


def get_test_train_indexes(df,i):

    IndexTrain = df.loc[df[i]].index
    IndexTest = df.index.difference(IndexTrain)

    return IndexTrain,IndexTest


def generate_CV_sets(cv_primary,base,id_list,nmois=None,replace=False,std=True):

    X = proc.import_X(base,nmois,std=std)
    Y = proc.import_Y(base,nmois)

    for id in id_list:

        df = pd.DataFrame(index=X.index,columns=np.arange(1,cv_primary.get_n_splits()))

        for i,(IndexTrain_iloc,IndexTest) in enumerate(cv_primary.split(X,Y)):

            IndexTrain = X.iloc[IndexTrain_iloc].index

            Index = [i in IndexTrain for i in X.index]
            df[i] = Index

        foldername = get_foldername('CV_primary',base)
        filename = get_filename(cv_primary,id,nmois,std)
        df.to_pickle(foldername + filename)


def Build_optimizer(hyperparameters,predicteur='auto',predicteur_params={},Hyperparameter_optimizer='auto',optimizer_params={},cv_nested=None):

    if predicteur in ['LogR2','auto']:
        predicteur = LogisticRegression(n_jobs=-1)
    elif predicteur == 'LogR1':
        predicteur = LogisticRegression(penalty='l1',n_jobs=-1)
    elif predicteur == 'LinR':
        predicteur = LinearRegression(n_jobs=-1)
    elif predicteur == 'Lasso':
        predicteur = Lasso(n_jobs=-1)
    elif predicteur == 'Ridge':
        predicteur = Ridge(n_jobs=-1)
    else:
        predicteur = predicteur

    predicteur.set_params(predicteur_params)

    if Hyperparameter_optimizer in ['auto','Grid']:
        optimizer = GridSearchCV(predicteur,hyperparameters,cv_nested=cv_nested)
    elif Hyperparameter_optimizer == 'Random':
        optimizer = RandomizedSearchCV(predicteur,hyperparameters,cv_nested=cv_nested)
    else:
        optimizer = Hyperparameter_optimizer(predicteur,hyperparameters,cv_nested=cv_nested)
    
    optimizer.set_params(optimizer_params)

    return optimizer


def score_model(self,base,id_list,optimizer,cv_primary,classifieur_frelevance,classifieur_fselect,metrics,nmois=None,std=True):

    param = optimizer.get_params()

    validation_metric = param['scoring'].__dict__['_score_func']

    MI = pd.MultiIndex.from_arrays([['general'],['fs_size']])
    
    arrays = [['Test'] * len(metrics) + 2,['score',validation_metric.__name__,*metrics.keys()]]
    MI2 = pd.MultiIndex.from_arrays(arrays)
    MI = MI.append(MI2)
    
    param_grid = param['param_grid']
    param_namelist = ['param_' + p for p in param_grid.keys()]
    arrays = [['Validation'] * len(param_grid),[*param_grid.keys()]]
    MI3 = pd.MultiIndex.from_arrays(arrays)
    MI = MI.append(MI3)

    foldername_cv,foldername_fs,foldername_scoring = get_foldername('all',base,classifieur_fr,classifieur_fs,classifieur_scored)

    os.makedirs(os.path.dirname(foldername_scoring), exist_ok=True)

    for id in id_list:

        filename = get_filename(cv_primary,id,nmois,std)

        df_cv = pd.read_pickle(foldername_cv + filename)
        df_fs = pd.read_pickle(foldername_fs + filename)

        X = pd.import_X(base,nmois,std)
        Y = pd.import_Y(base,nmois)

        for i in range(cv_primary.get_n_splits()):

            IndexVal_cv = df_cv[i]
            IndexVal_fs = df_fs[i].loc[X.columns]

            IndexTest_cv = X.index.difference(IndexVal_cv)

            Xval = X.loc[IndexVal_cv,IndexVal_fs]
            Yval = Y.loc[IndexVal_cv]

            Xtest = X.loc[IndexTest_cv,IndexVal_fs]
            Ytest = Y.loc[IndexTest_cv]

            optimizer.fit(Xval,Yval)

            best_params = GdS.best_params_

            score = optimizer.score(Xtest,Ytest)
            validation_metric = optimizer.best_score_


        df_output = pd.DataFrame(index=np.arange(cv_primary.get_n_splits()),columns=MI)

class ModelTester:

    def __init__(self,feature_selector,optimizer):

        self.optimizer = optimizer
        self.feature_selector = feature_selector

    # def test_model(self,base,id_list,nmois=None):


class FeatureSelector:

    def __init__(self,classifieur_frelevance,grid_frelevance,weight_frelevance,classifieur_fselect,metric_fselect,smoothing_fselect='auto',cv_frelevance=None,cv_fselect=None):

        if cv_frelevance is None:
            cv_frelevance = RepeatedKFold(n_splits=5,n_repeats=10)

        if cv_fselect is None:
            cv_fselect = KFold(n_splits=5,shuffle=True)

        self.classifieur_frelevance = classifieur_frelevance
        self.classifieur_fselect = classifieur_fselect

        self.grid_frelevance = grid_frelevance
        self.weight_frelevance = weight_frelevance

        self.cv_fselect = cv_fselect
        self.cv_frelevance = cv_frelevance

        self.metric_fselect = metric_fselect
        self.smoothing_fselect = smoothing_fselect

    def select_features(self,X,Y):

        relevance = self.classifieur_frelevance.feature_relevance(X=X,Y=Y,metrics={},cv=self.cv_frelevance,weights=[self.weight_frelevance])
        relevance.stat_percentage()
        percentage = relevance.data[self.weight_frelevance].loc['percentage']

#        print('done_frelevance')

        m,s,x = self.classifieur_fselect.stat_seuil(self.grid_frelevance,percentage,metrics={'loss':self.metric_fselect},cv=self.cv_fselect,X=X,Y=Y)
        moy,std = np.array(m['loss']),np.array(s['loss'])
        graph = {'abscisse':x,'moyenne':moy,'std':std}

#        print('done_stat_seuil')

        moy_smooth = self.smoothing_fselect(moy)
        compare = x[np.argmin(moy_smooth)] * np.ones(len(percentage))
        Index_bool = np.greater(percentage,compare)

        return Index_bool,graph

    def generate_featureselection(self,base,cv_primary,id_list,nmois=None,save_graph=False,**kwargs):

        std = kwargs.get('std',True)
        replace = kwargs.get('replace',False)

        foldername_input,foldername_output = get_foldername('FS+CV',base,self.classifieur_frelevance,self.classifieur_fselect)

        os.makedirs(os.path.dirname(foldername_output), exist_ok=True)

        # if nmois is None:
        #     predicteur_type = 'R'
        #     nmois_str = ''
        # else:
        #     predicteur_type = 'C'
        #     nmois_str = f'-{nmois}'

        # predicteur_str = predicteur_type + nmois_str

        for id in id_list:

            print(id)

            filename = get_filename(cv_primary,id,nmois,std=std)

            df_cv = pd.read_pickle(foldername_input + filename)

            X = proc.import_X(base,nmois)
            Y = proc.import_Y(base,nmois)

            output = pd.DataFrame(index=X.columns,columns=np.arange(cv_primary.get_n_splits()))

            graph_dict = {}

            for i in range(cv_primary.get_n_splits()):

                IndexTrain,IndexTest = get_test_train_indexes(df_cv,i)

                Xtrain = X.loc[IndexTrain]
                Ytrain = Y.loc[IndexTrain]

                I,graph = self.select_features(X=Xtrain,Y=Ytrain)

                output[i] = I

                if save_graph:
                    graph_dict[i] = graph
            
            if save_graph:
                with open(foldername_output + f'Graph.pkl', 'wb') as fp:
                    pickle.dump(graph_dict,fp,protocol=pickle.HIGHEST_PROTOCOL)

            output.to_pickle(foldername_output + filename)
