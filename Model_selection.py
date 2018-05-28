from sklearn.metrics import log_loss,make_scorer
from sklearn.model_selection import KFold,GridSearchCV
import Traitement as proc


def score_predicteur(X,Y,classifieur,classifieur_fs,weight_fs,grid_fs,params_grid,cv_main=KFold(n_splits=3,shuffle=True),cv_grid=KFold(n_splits=5,shuffle=True),smoothing_fs=lambda m:proc.smooth(m,window_len=5,window='hanning'),**kwargs):

    scoring = kwargs.get('scoring',make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1]))
    GdS = GridSearchCV(classifieur.objetSK,params_grid,scoring=scoring,n_jobs=1,cv=cv_grid,return_train_score=False)

    Index_results = ['score','log_loss'] + ['param_' + k for k in params_grid]
    results = pd.DataFrame(columns=Index_results)

    for IndexValidation,IndexTest in cv_main.split(X,Y):

        Xval,Yval = X.iloc[IndexValidation],Y.iloc[IndexValidation]
        Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]

        I = classifieur_fs.feature_select_from_model_weights(Xval,Yval,classifieur_fs,weight_fs,grid_fs,smoothing_fs)

        GdS.fit(Xval[I],Yval)
        estimator = GdS.best_estimator_
        data = GdS.best_params_
        
        data['loss'] = log_loss(Ytest,estimator.predict_proba(Xtest[I]),labels=[0,1])
        data['score'] = estimator.score(Xtest,Ytest)

        S = pd.Series(data=data,index=Index_results)
        results = results.append(S,ignore_index=True)
        
    return results
