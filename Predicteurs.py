from sklearn.model_selection import RepeatedKFold,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso,Ridge
from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd
import glob
import re
import Traitement as proc

PREDICTEUR_DICT = {
    
    'LinearRegression':'LinR',
    'Lasso':'LinR1',
    'Ridge':'LinR2',
    'LogisticRegression':'LogR',
    'GridSearchCV':'GdsCV',
    'XGBClassifier':'XGB',
    'XGBRegressor':'XGB',
    'RandomForestClassifier':'RF',
    'RandomForestRegressor':'RF',
    'LogisticRegressionCV':'LogRCV',
    'GridSearchCV':'GSCV',
    'RandomizedSearchCV':'RSCV'
}


def get_name_from_SK_predictor(SKpredictor):

    name = PREDICTEUR_DICT[type(SKpredictor).__name__]

    if name == 'LogR':
        name = name + SKpredictor.get_params()['penalty'][1]

    return name


def get_name_from_optimizer(opt):

    try:

        name = opt.name

    except AttributeError:

        name = get_name_from_SK_predictor(opt)

    return name


def seuil(Array,S):

    f_s = lambda x: int(x > S)
    f = np.vectorize(f_s)

    return f(Array)


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


class PredictorCV:

    def __init__(self,optimizer):

        self.optimizer = optimizer
        self.name = get_name_from_optimizer(optimizer)

    def fit(self,X,Y):

        self.optimizer.fit(X,Y)

    def predict_proba(self,X):

        return self.optimizer.predict_proba(X)

    def score(self,X,Y):

        return self.optimizer.score(X,Y)

    def get_params(self):

        return self.optimizer.get_params()

    def best_params(self):

        return self.optimizer.best_params_

    def best_score(self):

        return self.optimizer.best_score_


class Predictor:

    def __init__(self, predicteur):

        self.predicteur = predicteur
        self.name = get_name_from_SK_predictor(predicteur)


class Classifier(Predictor):

    def __init__(self, predicteur):

        Predictor.__init__(self, predicteur)

    def feature_relevance_XY(self,X,Y,metrics,weights,cv,**kwargs):

        index_weights = pd.MultiIndex.from_product([weights,X.columns.values],names=['weight','probe'])
        index_metrics = pd.MultiIndex.from_product([['metrics'],[*metrics.keys()]],names=['','probe'])
        index_df = index_metrics.append(index_weights)
        df = pd.DataFrame(columns=index_df)

        for IndexTrain,IndexTest in cv.split(X,Y):

            Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
            Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
            self.predicteur.fit(Xtrain,Ytrain)

            row = np.array([])

            for k,m in metrics.items():

                loss = m(Ytest,self.predicteur.predict_proba(Xtest),labels=[0,1])
                row = np.append(row,loss)

            for w in weights:

                row = np.append(row, getattr(self.predicteur,w))

            S = pd.Series(row,index_df)
            df = df.append(S,ignore_index=True)

        return Resultat(df)

    # def feature_select_percentage(self,X,Y,percentage,cv_fselect,metric,grid_fselect,f_smoothing):

    #     m,s,x = self.stat_seuil(X=X,Y=Y,metrics=metric,cv=cv_fselect,grid=grid_fselect,percentage=percentage)
    #     name = [*metric.keys()][0]
    #     moy,std = np.array(m[name]),np.array(s[name])

    #     moy_smooth = f_smoothing(moy)

    #     compare = x[np.argmin(moy_smooth)] * np.ones(len(percentage))
    #     index_percent = np.greater(percentage,compare)
    #     I = percentage.loc[index_percent].index

    #     return I,[m,s,x]

    def stat_seuil(self,X,Y,metrics,cv,grid,percentage,**kwargs):

        avg,var = dict.fromkeys(metrics.keys(),[]),dict.fromkeys(metrics.keys(),[])
        for i,k in enumerate(grid):
            compare = k * np.ones(len(percentage))
            index_percent = np.greater(percentage,compare)
            I = percentage.loc[index_percent].index
            Xk = X[I]
            if Xk.shape[1] == 0:
                grid = grid[:i - len(grid)]
                break
            Res = self.feature_relevance(X=Xk,Y=Y,weights=[],metrics=metrics,cv=cv)
            Res.calculate_m(metrics)
            Res.calculate_v(metrics)

            for (k,a),(k,s) in zip(avg.items(),var.items()):

                a = a.append(Res.moyennes[k])
                s = s.append(Res.variances[k])

        return avg,var,grid


class Regressor(Predictor):
    
    def __init__(self,predicteur,cv,X=None,Y=None,base=None,metrics={},weights=[]):
    
        Predictor.__init__(self,predicteur,cv,X,Y,base,metrics,weights)

    def feature_relevance(self,N=100,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        metrics = kwargs.get('metrics',self.metrics)
        weights = kwargs.get('weights',self.weights)
        info = kwargs.get('info',{'base':self.base,'predicteur':PREDICTEUR_DICT[type(self.predicteur).__name__],'predicteur_type':'R'})

        index_weights = pd.MultiIndex.from_product([weights,X.columns.values],names=['weight','probe'])
        index_metrics = pd.MultiIndex.from_product([['metrics'],[*metrics.values()]],names=['','probe'])
        index_df = index_metrics.append(index_weights)

        df = pd.DataFrame(columns=index_df)

        for k in range(N):

            kf = self.cv.split(X,Y)

            for i in range(self.cv.get_n_splits()):

                IndexTrain,IndexTest = next(kf)
                Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
                Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
                self.predicteur.fit(Xtrain,Ytrain)
                
                row = np.array([])

                for k,m in metrics.items():

                    loss = m(Ytest,self.predicteur.predict(Xtest))
                    row = np.append(row,loss)

                for w in weights:

                    row = np.append(row, getattr(self.predicteur,w))

                S = pd.Series(row,index_df)
                df = df.append(S,ignore_index=True)

        return Resultat(df,info=info)
    
    def stat_seuil(self,grid,percentage,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        N = kwargs.get('N',100)
        
        avg,var = [],[]
        for k in grid:
            Index_k = np.nonzero(seuil(percentage,k))[0]
            Xk = X.iloc[:,Index_k]
            a,v = self.stat_erreur_quadratique_moyenne(N=N,X=Xk,Y=Y)
            avg.append(a)
            var.append(v)
        
        return avg,var

class Resultat:

    def __init__(self,data):

        self.data = data

        if 'metrics' in self.data.columns:
            self.index_weights = self.data.columns.get_level_values(0).unique().drop('metrics').values
            self.index_metrics = self.data['metrics'].columns.values
            index = np.append(self.index_metrics,self.index_weights)
        else:
            index = self.data.columns.get_level_values(0).unique().values

        self.moyennes = dict.fromkeys(index,None)
        self.variances = dict.fromkeys(index,None)

        self.index_col = data.columns.values

    def calculate_m(self,mlist):

        # for m in mlist:

        #     if m in df['metrics'].values:

        #         self.moyennes[m] = np.mean(self.data['metrics',m])

        index_row = [CastableInt(i) for i in self.data.index.values]

        self.moyennes.update({m:np.mean(self.data['metrics',m].loc[index_row]) for m in mlist if m in self.data['metrics'].columns})

    def calculate_v(self,mlist):

        index_row = [CastableInt(i) for i in self.data.index.values]

        #self.calculate_m([m for m in mlist if (m in self.index_metrics) and (self.moyennes[m] is None)])

        self.variances.update({m: np.var(self.data['metrics',m].loc[index_row].values) for m in mlist if m in self.data['metrics'].columns})

        # for m in [m for m in mlist if m in self.index_metrics]:

        #     self.variances[m] = np.mean((self.data[m].values - self.moyennes[m])**2)

    def stat_m(self):

        index_row = [CastableInt(i) for i in self.data.index.values]

        row = [np.mean(self.data[i].loc[index_row]) for i in self.index_col]

        self.data.loc['moyenne'] = row

    def stat_v(self,redo_m=False):

        index_row = [CastableInt(i) for i in self.data.index.values]

        # if ('moyenne' not in self.data.index.values) or redo_m:

        #     self.stat_m()

        row = [np.var(self.data[j].loc[index_row]) for j in self.index_col]

        self.data.loc['variance'] = row

    def stat_percentage(self,threshold=0):

        index_row = [CastableInt(i) for i in self.data.index.values]
        row = np.zeros(len(self.index_col))

        f = np.vectorize(lambda x: int(abs(x) > threshold))

        for i,j in enumerate(self.index_col):

            row[i] = np.mean(f(self.data[j].loc[index_row].values))

        self.data.loc['percentage'] = row

    def save(self,filename,replace=False,**kwargs):

        # pattern = '.*_(\d*?).csv'

        # fname = rf'{self.info["base"]}/Stat.{self.info["predicteur_type"]}.{self.info["predicteur"]}'

        # if self.info['predicteur_type'] == 'C':
        #     fname = fname + f"-{self.info['nmois']}"

        # fnamelist = glob.glob(fname + '_*.csv')

        # values = [int(m[0]) for m in [re.findall(pattern,f) for f in fnamelist]]

        # if values == []:
        #     number = kwargs.get('number',0)
        # elif not replace:
        #     number = max(values) + 1
        # else:
        #     number = kwargs.get('number',max(values))
        
        self.data.to_csv(filename)


#   def add_rows(self):

    def add_parameter(self,weight_name,weight_list):

        self.weights[weight_name] = weight_list

    def add_metric(self,metric):

        self.metrics.append(metric)


def load_resultat(info={},filename="",**kwargs):

        if info == {}:

            data = pd.read_csv(filename,index_col=0,header=[0,1])

            info['base'],filename = filename.split('/')
            info['predicteur_type'],filename = filename.split('.')[1:-1]
            filename,info['id'] = filename.split('_')
            if (info['predicteur_type'] == 'C'):
                info['predicteur'],info['nmois'] = filename.split('-')
            else:
                info['predicteur'] = filename

        else:

            f = rf'{info["base"]}/Stat.{info["predicteur_type"]}.{info["predicteur"]}'

            if info['predicteur_type'] == 'C':
                f = f + f"-{info['nmois']}"

            f = f + f'_{info["id"]}.csv'

            data = pd.read_csv(f,index_col=0,header=[0,1],dtype={0:object})

        return Resultat(data,info=info)


def CastableInt(s):

    try:
        int(s)
        return True
    except ValueError:
        return False
