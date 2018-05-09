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

    def feature_relevance(self,N=100,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        metrics = kwargs.get('metrics',self.metrics)
        weights = kwargs.get('weights',self.weights)

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

                    loss = m(Ytest,self.objetSK.predict_proba(Xtest))
                    row = np.append(row,loss)

                for w in weights:

                    row = np.append(row, getattr(self.objetSK,w))

                S = pd.Series(row,index_df)
                df = df.append(S,ignore_index=True)

        return Resultat(df,info={'base':self.folder,'predicteur':PREDICTEUR_DICT[type(self.objetSK).__name__],'predicteur_type':'C','nmois':self.nmois})

    # def logloss(self,**kwargs):

    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
    #     kf = self.cv.split(X,Y)
    #     loss = 0

    #     for i in range(self.cv.get_n_splits()):
    #         IndexTrain,IndexTest = next(kf)
    #         Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
    #         Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
    #         self.objetSK.fit(Xtrain,Ytrain)
    #         loss += log_loss(Ytest,self.objetSK.predict_proba(Xtest))

    #     return loss / X.shape[0]

    # def stat_logloss(self,N=100,**kwargs):

    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
    #     l = []

    #     for i in range(N):
    #         l.append(self.logloss(X=X,Y=Y))
        
    #     moyenne = np.mean(l)
    #     variance = np.sum((l - moyenne)**2) / (len(l) - 1)
        
    #     return moyenne,variance

    # def score(self,**kwargs):
        
    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
        
    #     kf = self.cv.split(X,Y)
    #     score = 0
        
    #     for i in range(self.cv.get_n_splits()):
    #         IndexTrain,IndexTest = next(kf)
    #         Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
    #         Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
    #         self.objetSK.fit(Xtrain,Ytrain)
    #         score += self.objetSK.score(Xtest,Ytest)
        
    #     return score / self.cv.get_n_splits()

    # def stat_score(self,N=100,**kwargs):

    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
        
    #     l = []

    #     for i in range(N):
    #         l.append(self.score(X=X,Y=Y))
        
    #     moyenne = np.mean(l)
    #     variance = np.mean((l - moyenne)**2)
        
    #     return moyenne,variance

    def grid_search(self,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)

    def stat_seuil(self,grid,percentage,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        N = kwargs.get('N',100)
        skipped = 0
        avg,var = [],[]
        for i,k in enumerate(grid):
            Index_k = np.nonzero(seuil(percentage,k))[0]
            Xk = X.iloc[:,Index_k]
            if Xk.shape[1] == 0:
                skipped = len(grid) - i
                break
            a,v = self.stat_logloss(N=N,X=Xk,Y=Y)
            avg.append(a)
            var.append(v)
        
        return avg,var,skipped

    def load_X(self,folder,**kwargs):
        filename = kwargs.get('filename',rf'X_classification-{self.nmois}.csv')
        self.X = pd.read_csv(folder + filename,index_col=0)
        
    def load_Y(self,folder,**kwargs):
        filename = kwargs.get('filename',rf'Y_classification-{self.nmois}.csv')
        self.Y = pd.read_csvv(folder + filename,index_col=0,header=None)

    
class LinearClassifieur(GeneralClassifier):
    
    def __init__(self,objetSK,nmois,cv,X=None,Y=None,folder=None,**kwargs):

        weights = kwargs.get('weights',['coef_'])
        
        GeneralClassifier.__init__(self,objetSK,nmois,cv,X=X,Y=Y,folder=folder,weights=weights)
    
    # def feature_relevance(self,N,**kwargs):

    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)

    #     index_df = np.append(X.columns.values,'logloss')
    #     df = pd.DataFrame(columns=index_df)
        
    #     for k in range(N):
    #         kf = self.cv.split(X,Y)
    #         for i in range(self.cv.get_n_splits()):
    #             IndexTrain,IndexTest = next(kf)
    #             Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
    #             Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
    #             self.objetSK.fit(Xtrain,Ytrain)
    #             loss = log_loss(Ytest,self.objetSK.predict_proba(Xtest))
    #             row = np.append(self.objetSK.feature_importances_,loss)
    #             S = pd.Series(row,index_df)
    #             df = df.append(S,ignore_index=True)
        
    #     return Resultat(df,parameters={'feature_importances_':X.columns.values})


class EnsembleClassifieur(GeneralClassifier):

    def __init__(self,objetSK,nmois,cv,X=None,Y=None,folder=None):

        GeneralClassifier.__init__(self,objetSK,nmois,cv,X=X,Y=Y,folder=folder,metrics={'logloss':log_loss},weights=['feature_importances_'])

    # def feature_relevance(self,N=100,threshold=0,**kwargs):

    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)

    #     index_df = np.append(X.columns.values,'logloss')
    #     df = pd.DataFrame(columns=index_df)

    #     for k in range(N):
    #         kf = self.cv.split(X,Y)
    #         for i in range(self.cv.get_n_splits()):
    #             IndexTrain,IndexTest = next(kf)
    #             Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
    #             Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
    #             self.objetSK.fit(Xtrain,Ytrain)
    #             loss = log_loss(Ytest,self.objetSK.predict_proba(Xtest))
    #             row = np.append(self.objetSK.feature_importances_,loss)
    #             S = pd.Series(row,index_df)
    #             df = df.append(S,ignore_index=True)

    #     return df


class GeneralRegresser(Predicteur):
    
    def __init__(self,objetSK,cv,X=None,Y=None,folder=None,metrics={},weights=[]):
    
        Predicteur.__init__(self,objetSK,cv,X,Y,folder,metrics,weights)

    def feature_relevance(self,N=100,**kwargs):

        X = kwargs.get('X',self.X)
        Y = kwargs.get('Y',self.Y)
        metrics = kwargs.get('metrics',self.metrics)
        weights = kwargs.get('weights',self.weights)

        index_metrics = np.array([*metrics])
        index_weights = []
        index_df = np.array([])

        for w in weights:

            prefix = w + '_'
            index = prefix + X.columns.values
            index_weights.append(index)
            index_df = np.append(index_df, index)

        index_df = np.append(index_metrics,index_weights)

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

                    loss = m(Ytest,self.objetSK.predict(Xtest))
                    row = np.append(row,loss)

                for w in weights:

                    row = np.append(row, getattr(self.objetSK,w))

                S = pd.Series(row,index_df)
                df = df.append(S,ignore_index=True)

        return Resultat(df,metrics=[*metrics],weights=dict(zip(weights,index_weights)))

    # def erreur_quadratique_moyenne(self,**kwargs):
        
    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
        
    #     kf = self.cv.split(X,Y)
    #     loss = 0
        
    #     for i in range(self.cv.get_n_splits()):
            
    #         IndexTrain,IndexTest = next(kf)
    #         Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
    #         Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
    #         self.objetSK.fit(Xtrain,Ytrain)
    #         loss += mean_squared_error(Ytest,self.objetSK.predict(Xtest))
        
    #     return loss / X.shape[0]
    
    # def stat_erreur_quadratique_moyenne(self,N=100,**kwargs):
        
    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
    #     l = []
        
    #     for i in range(N):
    #         l.append(self.erreur_quadratique_moyenne(X=X,Y=Y))
        
    #     moyenne = np.mean(l)
    #     variance = np.mean((l - moyenne)**2)
        
    #     return moyenne,variance
        
    # def erreur_moyenne_absolue(self,**kwargs):
        
    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
        
    #     kf = self.cv.split(X,Y)
    #     loss = 0
        
    #     for i in range(self.cv.get_n_splits()):
            
    #         IndexTrain,IndexTest = next(kf)
    #         Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
    #         Xtest,Ytest = X.iloc[IndexTest],Y.iloc[IndexTest]
    #         self.objetSK.fit(Xtrain,Ytrain)
    #         loss += mean_absolute_error(Ytest,self.objetSK.predict(Xtest))
        
    #     return loss / X.shape[0]
    
    # def stat_erreur_moyenne_absolue(self,N=100,**kwargs):
        
    #     X = kwargs.get('X',self.X)
    #     Y = kwargs.get('Y',self.Y)
    #     l = []
        
    #     for i in range(N):
    #         l.append(self.erreur_moyenne_absolue(X=X,Y=Y))
        
    #     moyenne = np.mean(l)
    #     variance = np.mean((l - moyenne)**2)
        
    #     return moyenne,variance
    
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

        self.moyennes.update({m:np.mean(self.data['metrics'],m) for m in mlist if m in self.data['metrics'].columns})

    def calculate_v(self,mlist):

        self.calculate_m([m for m in mlist if (m in self.index_metrics) and (self.moyennes[m] is None)])

        self.variances.update({m: np.mean((self.data[m].values - self.moyennes[m])**2) for m in mlist if m in self.index_metrics})

        # for m in [m for m in mlist if m in self.index_metrics]:

        #     self.variances[m] = np.mean((self.data[m].values - self.moyennes[m])**2)

    def stat_m(self):

        index_row = [True if type(i) is int else False for i in self.data.index.values]

        row = [np.mean(self.data[i].loc[index_row]) for i in self.index_col]
        S = pd.Series(row)
        self.data.loc['moyenne'] = S

    def stat_v(self,redo_m=False):

        index_row = [True if type(i) is int else False for i in self.data.index.values]

        if ('moyenne' not in self.data.index.values) or redo_m:

            self.stat_m()

        row = [(np.mean(self.data[i].loc[index_row] - self.data.loc['moyenne'].values[j]))**2 for i,j in enumerate(self.index_col)]
        S = pd.Series(row)

        self.data.loc['variance'] = S

    def stat_percentage(self,threshold=0):

        index_row = [True if type(i) is int else False for i in self.data.index.values]
        row = np.zeros(len(self.index_col))
        compare = threshold * np.ones(len(self.data[self.index[0].loc[index_row]]))

        row = [np.mean(np.greater(self.data[i].loc[index_row].values,compare) + 0) for i in self.index]

        S = pd.Series(row)
        self.data.loc['percentage'] = S

    def save(self,replace=False,**kwargs):

        pattern = '.*_(\d*?).csv'

        fname = rf'{self.info["base"]}/Stat.{self.info["predicteur_type"]}.{self.info["predicteur"]}'

        if self.info['predicteur_type'] == 'C':
            fname = fname + f"-{self.info['nmois']}"

        fnamelist = glob.glob(fname + '_*.csv')

        values = [int(m[0]) for m in [re.findall(pattern,f) for f in fnamelist]]

        number = kwargs.get('number',self.info['id'])
        
        if number in values and (not replace):
            number = max(values) + 1

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

            data = pd.read_csv(f,index_col=0,header=[0,1])

        return Resultat(data,info=info)
