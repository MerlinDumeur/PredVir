from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd
import glob
import re

PREDICTEUR_DICT = {
    
    'LinearRegression':'Lin',
    'LogisticRegression':'Log'}

PREDICTEUR_DICT['XGBClassifier'] = PREDICTEUR_DICT['XGBRegressor'] = 'XGB'
PREDICTEUR_DICT['RandomForestClassifier'] = PREDICTEUR_DICT['RandomForestRegressor'] = 'RF'


class Predicteur:

    def __init__(self, objetSK, cv, X=None, Y=None, folder=None, metrics={}, weights=[]):

        self.cv = cv
        self.objetSK = objetSK
        self.X = X
        self.Y = Y
        self.metrics = metrics
        self.weights = weights
        self.folder = folder

        if ((X is None) and (folder is not None)):
            self.load_X(folder)
        if ((Y is None) and (folder is not None)):
            self.load_Y(folder)

    def set_objetSK(self,objetSK):
        self.objetSK = objetSK

    def set_cv(self,cv):
        self.cv = cv


class GeneralClassifier(Predicteur):

    def __init__(self, objetSK, nmois, cv, X=None, Y=None, folder=None, metrics={'logloss':log_loss}, weights=[]):

        self.nmois = nmois
        Predicteur.__init__(self, objetSK, cv, X, Y, folder,metrics,weights)

    def feature_relevance(self,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        metrics = kwargs.get('metrics',self.metrics)
        weights = kwargs.get('weights',self.weights)
        info = kwargs.get('info',{'base':self.folder,'predicteur':PREDICTEUR_DICT[type(self.objetSK).__name__],'predicteur_type':'C','nmois':self.nmois})

        index_weights = pd.MultiIndex.from_product([weights,X.columns.values],names=['weight','probe'])
        index_metrics = pd.MultiIndex.from_product([['metrics'],[*metrics.keys()]],names=['','probe'])
        index_df = index_metrics.append(index_weights)

        df = pd.DataFrame(columns=index_df)

        for IndexTrain,IndexTest in self.cv.split(X,Y):

            Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
            Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
            self.objetSK.fit(Xtrain,Ytrain)

            row = np.array([])

            for k,m in metrics.items():

                loss = m(Ytest,self.objetSK.predict_proba(Xtest),labels=[0,1]) / Xtest.shape[0]
                row = np.append(row,loss)

                row = np.array([m(Ytest,self.objetSK.predict_proba(Xtest)) for m in metrics.values()])

                for w in weights:

                    row = np.append(row, getattr(self.objetSK,w))

                    S = pd.Series(row,index_df)
                    df = df.append(S,ignore_index=True)

        return Resultat(df,info=info)

    def grid_search(self,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)

    def stat_seuil(self,grid,percentage,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        avg,var = dict.fromkeys(self.metrics.keys(),[]),dict.fromkeys(self.metrics.keys(),[])
        for i,k in enumerate(grid):
            compare = k * np.ones(len(percentage))
            index_percent = np.greater(percentage,compare)
            I = percentage.loc[index_percent].index
            Xk = X[I]
            if Xk.shape[1] == 0:
                grid = grid[:i - len(grid)]
                break
            Res = self.feature_relevance(X=Xk,Y=Y)
            Res.calculate_v(self.metrics)

            for (k,a),(k,s) in zip(avg.items(),var.items()):

                a = a.append(Res.moyennes[k])
                s = s.append(Res.variances[k])
            
#           var.append(Res.variances[self.metrics[0]])
#           a,v = self.stat_logloss(N=N,X=Xk,Y=Y)

        return avg,var,grid

    def load_X(self,folder,**kwargs):
        filename = kwargs.get('filename',rf'X_classification-{self.nmois}.csv')
        self.X = pd.read_csv(folder + filename,index_col=0)
        
    def load_Y(self,folder,**kwargs):
        filename = kwargs.get('filename',rf'Y_classification-{self.nmois}.csv')
        self.Y = pd.read_csvv(folder + filename,index_col=0,header=None)

    
class LinearClassifieur(GeneralClassifier):
    
    def __init__(self,objetSK,nmois,cv,X=None,Y=None,folder=None,**kwargs):

        weights = kwargs.get('weights',['coef_'])
        metrics = kwargs.get('metrics',{'logloss':log_loss})
        
        GeneralClassifier.__init__(self,objetSK,nmois,cv,X=X,Y=Y,folder=folder,weights=weights,metrics=metrics)


class EnsembleClassifieur(GeneralClassifier):

    def __init__(self,objetSK,nmois,cv,X=None,Y=None,folder=None,**kwargs):

        weights = kwargs.get('weights',['feature_importances_'])
        metrics = kwargs.get('metrics',{'logloss':log_loss})

        GeneralClassifier.__init__(self,objetSK,nmois,cv,X=X,Y=Y,folder=folder,metrics=metrics,weights=weights)


class GeneralRegresser(Predicteur):
    
    def __init__(self,objetSK,cv,X=None,Y=None,folder=None,metrics={},weights=[]):
    
        Predicteur.__init__(self,objetSK,cv,X,Y,folder,metrics,weights)

    def feature_relevance(self,N=100,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        metrics = kwargs.get('metrics',self.metrics)
        weights = kwargs.get('weights',self.weights)
        info = kwargs.get('info',{'base':self.folder,'predicteur':PREDICTEUR_DICT[type(self.objetSK).__name__],'predicteur_type':'R'})

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
                self.objetSK.fit(Xtrain,Ytrain)
                
                row = np.array([])

                for k,m in metrics.items():

                    loss = m(Ytest,self.objetSK.predict(Xtest)) / Xtest.shape[0]
                    row = np.append(row,loss)

                for w in weights:

                    row = np.append(row, getattr(self.objetSK,w))

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

    def load_X(self,folder,**kwargs):
        filename = kwargs.get('filename',rf'X_regression.csv')
        self.X = pd.read_csv(folder + filename,index_col=0)
        
    def load_Y(self,folder,**kwargs):
        filename = kwargs.get('filename',rf'Y_regression.csv')
        self.Y = pd.read_csvv(folder + filename,index_col=0,header=None)


class LinearRegresser(GeneralRegresser):

    def __init__(self,objetSK,cv,X=None,Y=None,folder=None,metrics={'eqm':mean_squared_error,'eam':mean_absolute_error},weights=['coef_'],info={'type':'Classifieur','Modele':'Lineaire',}):

        GeneralRegresser.__init__(self,objetSK,cv,X,Y,folder,metrics,weights)

#     def feature_relevance(self,threshold,N=100,**kwargs):
        
#         X = kwargs.get('X',self.X)
#         Y = kwargs.get('Y',self.Y)
        
#         percent,avg = np.zeros(X.shape[1]),np.zeros(X.shape[1])
#         threshold = threshold * np.ones(X.shape[1])
        
#         n = self.cv.get_n_splits() * N
        
#         for k in range(N):
            
#             kf = self.cv.split(X,Y)
            
#             for i in range(self.cv.get_n_splits()):
                
#                 IndexTrain,IndexTest = next(kf)
#                 Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
#                 self.objetSK.fit(Xtrain,Ytrain)
#                 avg = avg + self.objetSK.coef_ / n
#                 index = np.greater(self.objetSK.coef_,threshold)
#                 print(index)
#                 percent[index[0]] += 1 / n
                
#         return percent,avg


def seuil(Array,S):
    f_s = lambda x: int(x > S)
    f = np.vectorize(f_s)
    return f(Array)


class Resultat:

    def __init__(self,data,info={}):

        self.data = data
        self.info = info

        self.index_weights = self.data.columns.get_level_values(0).unique().drop('metrics').values
        self.index_metrics = self.data['metrics'].columns.values
        index = np.append(self.index_metrics,self.index_weights)
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

        self.calculate_m([m for m in mlist if (m in self.index_metrics) and (self.moyennes[m] is None)])

        self.variances.update({m: np.mean((self.data['metrics',m].loc[index_row].values - self.moyennes[m])**2) for m in mlist if m in self.data['metrics'].columns})

        # for m in [m for m in mlist if m in self.index_metrics]:

        #     self.variances[m] = np.mean((self.data[m].values - self.moyennes[m])**2)

    def stat_m(self):

        index_row = [CastableInt(i) for i in self.data.index.values]

        row = [np.mean(self.data[i].loc[index_row]) for i in self.index_col]

        self.data.loc['moyenne'] = row

    def stat_v(self,redo_m=False):

        index_row = [CastableInt(i) for i in self.data.index.values]

        if ('moyenne' not in self.data.index.values) or redo_m:

            self.stat_m()

        row = [(np.mean(self.data[j].loc[index_row] - self.data.loc['moyenne'].values[i]))**2 for i,j in enumerate(self.index_col)]

        self.data.loc['variance'] = row

    def stat_percentage(self,threshold=0):

        index_row = [CastableInt(i) for i in self.data.index.values]
        row = np.zeros(len(self.index_col))

        f = np.vectorize(lambda x: int(x>threshold))

        for i,j in enumerate(self.index_col):

            row[i] = np.mean(f(self.data[j].loc[index_row].values))

        self.data.loc['percentage'] = row

    def save(self,replace=False,**kwargs):

        pattern = '.*_(\d*?).csv'

        fname = rf'{self.info["base"]}/Stat.{self.info["predicteur_type"]}.{self.info["predicteur"]}'

        if self.info['predicteur_type'] == 'C':
            fname = fname + f"-{self.info['nmois']}"

        fnamelist = glob.glob(fname + '_*.csv')

        values = [int(m[0]) for m in [re.findall(pattern,f) for f in fnamelist]]

        if values == []:
            number = kwargs.get('number',0)
        elif not replace:
            number = max(values) + 1
        else:
            number = kwargs.get('number',max(values))
        
        filename = kwargs.get('filename',fname + f'_{number}.csv')
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
