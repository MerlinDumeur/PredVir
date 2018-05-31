from sklearn.metrics import log_loss,make_scorer
from sklearn.model_selection import RepeatedKFold,KFold,GridSearchCV
import Traitement as proc
import pandas as pd
import pickle
import os

CV_dict = {
    'KFold':'KF',
    'RepeatedKFold':'RKF',
    'StratifiedKFold':'SKF',
    'RepeatedStratifiedKFold':'SRKF'
}


def score_predicteur(X,Y,classifieur,classifieur_frelevance,classifieur_fselect,weight_fs,grid_fs,params_grid,metric_fs=log_loss,cv_main=KFold(n_splits=3,shuffle=True),cv_grid=KFold(n_splits=5,shuffle=True),smoothing_fs=lambda m:proc.smooth(m,window_len=5,window='hanning'),**kwargs):

    scoring = kwargs.get('scoring',make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1]))
    metrics_test = kwargs.get('metrics',{'log_loss':log_loss})

    GdS = GridSearchCV(classifieur.objetSK,params_grid,scoring=scoring,n_jobs=-1,cv=cv_grid,return_train_score=False)

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


def get_folder_data(base,CV,id,nmois=None):

    predicteur_str = 'R' if nmois is None else f'C-{nmois}'

    return base + rf'/{predicteur_str}.{CV_dict[type(CV).__name__]}.{CV.get_n_splits()}-{id}/'


def generate_CV_sets(CV,base,id_list,nmois=None,replace=False):

    X = proc.import_X(base,nmois)
    Y = proc.import_Y(base,nmois)

    for id in id_list:

        folderdata = get_folder_data(base,CV,id,nmois)
        os.makedirs(os.path.dirname(folderdata), exist_ok=True)

        for i,(IndexTrain,IndexTest) in enumerate(CV.split(X,Y)):

            IndexTrain.to_frame().to_pickle(folderdata + f'IndexTrain-{i}.pkl')
            IndexTest.to_frame().to_pickle(folderdata + f'IndexTest-{i}.pkl')


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

        relevance = self.classifieur_frelevance(X=X,Y=Y,metrics={})
        relevance.stat_percentage()
        percentage = relevance.data[self.weight_frelevance].loc['percentage']

        m,s,x = self.stat_seuil(self.grid_frelevance,percentage,metrics={'loss':self.metric_fselect})
        moy,std = np.array(m['loss'])
        graph = {'abscisse':x,'moyenne':moy,'std':std}

        moy_smooth = self.smoothing_fselect(moy)
        compare = x[np.argmin(moy_smooth)] * np.ones(len(percentage))
        index_percent = np.greater(percentage,compare)

        I = percentage.loc[index_percent].index

        return I,graph

    def generate_featureselection(self,base,CV,id_list,nmois=None,save_graph=False):

        # if nmois is None:
        #     predicteur_type = 'R'
        #     nmois_str = ''
        # else:
        #     predicteur_type = 'C'
        #     nmois_str = f'-{nmois}'

        # predicteur_str = predicteur_type + nmois_str

        for id in id_list:

            folderdata = get_folder_data(base,CV,id)
            folderoutput = folderdata + rf'{self.classifieur_frelevance.name}-{self.classifieur_fselect.name}/'

            X = proc.import_X(base,nmois)
            Y = proc.import_Y(base,nmois)

            for i in range(CV.get_n_splits()):

                IndexTrain = pd.read_pickle(folderdata + rf'IndexTrain-{i}.pkl').index

                Xtrain = X.loc[IndexTrain]
                Ytrain = Y.loc[IndexTrain]

                I,graph = self.select_features(X=Xtrain,Y=Ytrain)

                I.to_frame().to_pickle(folderoutput + f'Index-{i}.pkl')

                if save_graph:
                    with open(folderoutput + f'Graph-{i}.pkl', 'wb') as fp:
                        pickle.dump(graph,fp,protocol=pickle.HIGHEST_PROTOCOL)
