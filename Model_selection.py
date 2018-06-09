from sklearn.metrics import log_loss,make_scorer,get_scorer
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

# def score_predicteur(X,Y,classifieur,classifieur_fr,classifieur_fs,weight_fs,grid_fs,params_grid,metric_fs=log_loss,cv_main=KFold(n_splits=3,shuffle=True),cv_grid=KFold(n_splits=5,shuffle=True),smoothing_fs=lambda m:proc.smooth(m,window_len=5,window='hanning'),**kwargs):

#     scoring = kwargs.get('scoring',make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1]))
#     metrics_test = kwargs.get('metrics',{'log_loss':log_loss})

#     GdS = GridSearchCV(classifieur.predicteur,params_grid,scoring=scoring,n_jobs=-1,cv=cv_grid,return_train_score=False)

#     msx_list = []

#     Index_results = ['score','log_loss','fs_size'] + [k for k in params_grid]
#     results = pd.DataFrame(columns=Index_results)

#     for IndexValidation,IndexTest in cv_main.split(X,Y):

#         Xval,Yval = X.iloc[IndexValidation],Y.iloc[IndexValidation]
#         Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]


#         I,msx = classifieur_fs.feature_select_percentage(X=Xval,Y=Yval,percentage=percentage,cv_fs=,classifieur_fr,,weight_fs,grid_fs,smoothing_fs)
#         X,Y,percentage,cv_fs,metric,grid_fs,f_smoothing

#         msx_list.append(msx)

#         GdS.fit(Xval[I],Yval)
#         estimator = GdS.best_estimator_
#         data = GdS.best_params_.copy()

#         data['log_loss'] = log_loss(Ytest,estimator.predict_proba(Xtest[I]),labels=[0,1])
#         data['score'] = estimator.score(Xtest[I],Ytest)
#         data['fs_size'] = len(I)

#         S = pd.Series(data=data,index=Index_results)

#         results = results.append(S,ignore_index=True)
        
#     return results,msx_list


def get_foldername(type,base,classifieur_fr=None,classifieur_fs=None,model=None,**kwargs):

    filename1 = base + rf'/cv-sets/'

    if type == 'CV_primary':

        return filename1

    filename2 = filename1 + rf'{classifieur_fr.name}-{classifieur_fs.name}/'

    if type == 'FS':

        return filename2

    elif type == 'FS+CV':

        return filename1,filename2

    filename3 = filename2 + rf'{model.name}/'

    if type == 'model_scoring':

        return filename3

    elif type == 'all':

        return filename1,filename2,filename3


def get_test_train_indexes(df,i):

    IndexTrain = df.loc[df[i]].index
    IndexTest = df.index.difference(IndexTrain)

    return IndexTrain,IndexTest


def generate_CV_sets(dataSet,replace=False):

    X = proc.import_X(dataSet.base,dataSet.nmois,std=dataSet.std)
    Y = proc.import_Y(dataSet.base,dataSet.nmois)

    for id in dataSet.id_list:

        df = pd.DataFrame(index=X.index,columns=np.arange(1,dataSet.cv_primary.get_n_splits()))

        for i,(IndexTrain_iloc,IndexTest) in enumerate(dataSet.cv_primary.split(X,Y)):

            IndexTrain = X.iloc[IndexTrain_iloc].index

            Index = [i in IndexTrain for i in X.index]
            df[i] = Index

        foldername = get_foldername('CV_primary',dataSet.base)
        filename = dataSet.get_filename(id)
        df.to_pickle(foldername + filename)


class DataSet:

    def __init__(self,base,id_list,cv_primary,nmois=None,std=True):

        self.base = base
        self.id_list = id_list
        self.cv_primary = cv_primary
        self.nmois = nmois
        self.std = std

    def get_filename(self,id):

        predicteur_str = 'R' if self.nmois is None else f'C-{self.nmois}'
        
        filename = rf'{predicteur_str}.{"S" if self.std else ""}.{CV_dict[type(self.cv_primary).__name__]}.{self.cv_primary.get_n_splits()}-{id}.pkl'

        return filename

    def import_X(self):

        return proc.import_X(self.base,self.nmois,self.std)

    def import_Y(self):

        return proc.import_Y(self.base,self.nmois)


class Model:

    def __init__(self,feature_selector,predictorCV):

        self.predictorCV = predictorCV
        self.feature_selector = feature_selector

    def score_model(self,dataSet,metrics={},use_fs=True):

        param = self.predictorCV.get_params()

        # if isinstance(param['scoring'],str):
        #     validation_metric = get_scorer(param['scoring'])
        # else:
        #     validation_metric = param['scoring'].__dict__['_score_func']

        MI = pd.MultiIndex.from_arrays([['general'],['fs_size']])
        MI_final = MI.copy()

        arrays_m = [['Test'] * (len(metrics) + 1),['score',*metrics.keys()]]
        MI2 = pd.MultiIndex.from_arrays(arrays_m)
        MI_final = MI_final.append(MI2)
        
        param_grid = param[self.predictorCV.param_grid_name]
        # param_namelist = ['param_' + p for p in param_grid.keys()]
        arrays_p = [['Validation'] * (len(param_grid) + 1),['score',*param_grid]]
        MI3 = pd.MultiIndex.from_arrays(arrays_p)
        MI_final = MI_final.append(MI3)

        foldername_cv,foldername_fs,foldername_scoring = get_foldername('all',dataSet.base,self.feature_selector.classifieur_fr,self.feature_selector.classifieur_fs,self.predictorCV)

        os.makedirs(os.path.dirname(foldername_scoring), exist_ok=True)

        X = dataSet.import_X()
        Y = dataSet.import_Y()

        for id in dataSet.id_list:

            filename = dataSet.get_filename(id)

            df_cv = pd.read_pickle(foldername_cv + filename)
            df_fs = pd.read_pickle(foldername_fs + filename)

            df_output = pd.DataFrame(index=np.arange(dataSet.cv_primary.get_n_splits()),columns=MI_final)

            for i in range(dataSet.cv_primary.get_n_splits()):

                IndexVal_fs = df_fs[i].loc[X.columns]
                IndexVal_cv,IndexTest_cv = get_test_train_indexes(df_cv,i)

                if use_fs:
                    Xval = X.loc[IndexVal_cv,IndexVal_fs]
                    Xtest = X.loc[IndexTest_cv,IndexVal_fs]
                else:
                    Xval = X.loc[IndexVal_cv]
                    Xtest = X.loc[IndexTest_cv]

                Yval = Y.loc[IndexVal_cv]
                Ytest = Y.loc[IndexTest_cv]

                self.predictorCV.fit(Xval,Yval)

                best_params = self.predictorCV.best_params()
                score = self.predictorCV.score(Xtest,Ytest)
                # vm = validation_metric(self.predictorCV,Xtest,Ytest)
    
                df_output.loc[i,'general'] = Xval.shape[1]
                # print(df_output.loc[i,'Test'])
                # print(score)
                # print(vm)
                # print([score,vm] + [metrics[m](Ytest,self.predictorCV.predict_proba(Xtest),labels=[0,1]) for m in arrays_m[1]])
                df_output.loc[i,'Test'] = [score] + [metrics[m](Ytest,self.predictorCV.predict_proba(Xtest),labels=[0,1]) for m in arrays_m[1][1:]]
                df_output.loc[i,'Validation'] = [self.predictorCV.best_score()] + [best_params[k] for k in arrays_p[1][1:]]
                # else:
                #     df_output.loc[i,'Validation'] = [self.predictorCV.best_score()] + [best_params[0]]

                # validation_metric = self.predictorCV.best_score()

            df_output.to_pickle(foldername_scoring + filename)


class FeatureSelector:

    def __init__(self,classifieur_fr,grid_fr,weight_fr,classifieur_fs,metric_fs,smoothing_fs='auto',cv_fr=None,cv_fs=None):

        if cv_fr is None:
            cv_fr = RepeatedKFold(n_splits=5,n_repeats=10)

        if cv_fs is None:
            cv_fs = KFold(n_splits=5,shuffle=True)

        self.classifieur_fr = classifieur_fr
        self.classifieur_fs = classifieur_fs

        self.grid_fr = grid_fr
        self.weight_fr = weight_fr

        self.cv_fs = cv_fs
        self.cv_fr = cv_fr

        self.metric_fs = metric_fs
        self.smoothing_fs = smoothing_fs

    def select_features(self,X,Y):

        relevance = self.classifieur_fr.feature_relevance_XY(X=X,Y=Y,metrics={},weights=[self.weight_fr],cv=self.cv_fr)
        relevance.stat_percentage()
        percentage = relevance.data[self.weight_fr].loc['percentage']

        m,s,x = self.classifieur_fs.stat_seuil(X=X,Y=Y,metrics={'loss':self.metric_fs},cv=self.cv_fs,grid=self.grid_fr,percentage=percentage)
        moy,std = np.array(m['loss']),np.array(s['loss'])
        graph = {'abscisse':x,'moyenne':moy,'std':std}

        moy_smooth = self.smoothing_fs(moy)
        compare = x[np.argmin(moy_smooth)] * np.ones(len(percentage))
        Index_bool = np.greater(percentage,compare)

        return Index_bool,graph

    def generate_featureselection(self,dataSet,save_graph=False,**kwargs):

        replace = kwargs.get('replace',False)

        foldername_input,foldername_output = get_foldername('FS+CV',dataSet.base,self.classifieur_fr,self.classifieur_fs)

        graphs = {}

        os.makedirs(os.path.dirname(foldername_output), exist_ok=True)

        # if nmois is None:
        #     predicteur_type = 'R'
        #     nmois_str = ''
        # else:
        #     predicteur_type = 'C'
        #     nmois_str = f'-{nmois}'

        # predicteur_str = predicteur_type + nmois_str

        for id in dataSet.id_list:

            print(id)

            filename = dataSet.get_filename(id)

            df_cv = pd.read_pickle(foldername_input + filename)

            X = dataSet.import_X()
            Y = dataSet.import_Y()

            output = pd.DataFrame(index=X.columns,columns=np.arange(dataSet.cv_primary.get_n_splits()))

            graph_dict = {}

            for i in range(dataSet.cv_primary.get_n_splits()):

                IndexTrain,IndexTest = get_test_train_indexes(df_cv,i)

                Xtrain = X.loc[IndexTrain]
                Ytrain = Y.loc[IndexTrain]

                I,graph = self.select_features(X=Xtrain,Y=Ytrain)

                output[i] = I

                if save_graph:
                    graph_dict[i] = graph

            output.to_pickle(foldername_output + filename)
            if save_graph:
                graphs[id] = graph_dict

        if save_graph:
            with open(foldername_output + f'Graph.pkl', 'wb') as fp:
                pickle.dump(graph_dict,fp,protocol=pickle.HIGHEST_PROTOCOL)
