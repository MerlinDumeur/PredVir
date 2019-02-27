import pandas as pd
import numpy as np


class GeneSelector:

    def __init__(self,model):

        self.model = model

    def returns_index(self):

        return True


class GeneSelector_GLM(GeneSelector):

    def __init__(self,model,passthrough=False):

        GeneSelector.__init__(self,model)
        self.passthrough = passthrough

    def select_genes(self,X,Y):

        # print(X)
        # print(Y)

        self.model.fit(X,Y)

        if 'CV' in self.model.model.__class__.__name__:

            coef = self.model.best_estimator_.coef_

        else:

            coef = self.model.coef_

        index_keep = np.greater(coef,np.zeros(len(coef)))

        if self.passthrough:

            return X.columns

        else:

            return index_keep[0]

    def serialize(self):

        return self.model.get_modelhash()


class GeneSelector_GLM_Iterative(GeneSelector):

    def __init__(self,model,n_limit,passthrough=False):

        GeneSelector.__init__(self,model)
        self.passthrough = passthrough
        self.n_limit = n_limit

    def select_genes(self,X,Y):

        # print(X)
        # print(Y)

        if self.passthrough:

            return X.columns

        kept_out = pd.Index([])
        loop = True

        while loop:

            Xtemp = X.loc[:,X.columns.difference(kept_out)]

            self.model.fit(Xtemp,Y)

            if 'CV' in self.model.model.__class__.__name__:

                coef = self.model.best_estimator_.coef_

            else:

                coef = self.model.coef_

            index_keep = np.greater(coef,np.zeros(len(coef)))[0]

            print(Xtemp.loc[:,index_keep].shape)

            if Xtemp.loc[:,index_keep].shape[0] + len(kept_out) > self.n_limit:

                loop = False

            else:

                kept_out = kept_out.append(Xtemp.loc[:,index_keep].columns)

        return kept_out

    def serialize(self):

        return self.model.get_modelhash() << 1


class GeneSelector_Participative:

    def __init__(self,model_fr,model_fs,cv_fr,cv_fs,metric_fs):

        self.model_fr = model_fr
        self.model_fs = model_fs

        self.cv_fr = cv_fr
        self.cv_fs = cv_fs

        self.metric_fs = metric_fs

    def select_genes(self,X,Y,weight_name,grid,savename=None):

        S = pd.Series(data=np.zeros(len(X.columns)),index=X.columns)

        for IndexTrain,IndexTest in self.cv_fr.split(X,Y):

            Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
            self.model_fr.fit(Xtrain,Ytrain)
            c = getattr(self.model_fr.best_estimator_,weight_name)[0]
            Idx = X.columns[c != 0]
            S[Idx] += 1

        S /= self.cv_fr.get_n_splits()

        avg = pd.Series(index=grid)
        # std = pd.Series(index=grid)

        for i,g in enumerate(grid):

            Xk = X.loc[:,S > g]
            
            if Xk.shape[1] == 0:
                grid = grid[:i - len(grid)]
                break

            temp_scores = []

            for IndexTrain,IndexTest in self.cv_fs.split(Xk,Y):

                Xtrain,Ytrain = Xk.iloc[IndexTrain],Y.iloc[IndexTrain]
                Xtest,Ytest = Xk.iloc[IndexTest],Y.iloc[IndexTest]

                self.model_fs.fit(Xtrain,Ytrain)

                temp_scores.append(self.metric_fs(Ytest,self.model_fs.predict_proba(Xtest),labels=[0,1]))

            avg[g] = np.mean(temp_scores)
            # std[g]

        if savename is not None:
            avg.to_pickle(savename)

        idx = avg.idxmin()

        return S > idx

    def serialize(self):

        return self.model_fr.get_modelhash() + self.model_fs.get_modelhash()


class GeneSelector_Participative2:

    def __init__(self,model_fr,cv_fr,n_keep):

        self.model_fr = model_fr
        self.cv_fr = cv_fr

        self.n_keep = n_keep

    def select_genes(self,X,Y,weight_name,savename=None):

        S = pd.Series(data=np.zeros(len(X.columns)),index=X.columns)

        for IndexTrain,IndexTest in self.cv_fr.split(X,Y):

            Xtrain,Ytrain = X.iloc[IndexTrain],Y.iloc[IndexTrain]
            self.model_fr.fit(Xtrain,Ytrain)
            c = getattr(self.model_fr.best_estimator_,weight_name)[0]
            Idx = X.columns[c != 0]
            S[Idx] += 1

        S /= self.cv_fr.get_n_splits()

        return S.nlargest(self.n_keep).index

    def serialize(self):

        return self.model_fr.get_modelhash() << 2


class GeneSelectorFile:

    def from_CVFILE(cvfile,GS,passthrough=False):

        return GeneSelectorFile(cvfile.filename[:-4] + str(GS.serialize()) + '.pkl',GS.serialize(),passthrough)

    def __init__(self,filename,hashcode,passthrough=False):

        self.passthrough = passthrough
        self.hash = 'PASSTHROUGH' if passthrough else hashcode
        self.df = pd.read_pickle(filename)

    def select_genes(self,i):

        idx = self.df.loc[:,i]
        # print(idx)
        # print(idx.shape)
        # print(self.df.loc[idx].values.shape)

        if self.passthrough:

            return self.df.index

        else:

            return (idx == 1).values

    def generate_file(dataset,cvfile,geneselector,**kwargs):

        df = pd.DataFrame(index=dataset.X.columns)

        for i,(Xtrain,Ytrain,Xtest,Ytest) in enumerate(dataset.CV_split(cvfile)):

            idkeep = geneselector.select_genes(Xtrain,Ytrain,**kwargs)
            df.loc[:,i] = 0
            df.loc[idkeep,i] = 1

        df.to_pickle(cvfile.filename[:-4] + str(geneselector.serialize()) + '.pkl')

    def returns_index(self):

        return True


class VariableCreator:

    def returns_index(self):

        return False


class VariableCreator_ACP(VariableCreator):

    def __init__(self,ACP):

        self.ACP = ACP

    def generate_train(self,X):

        return self.ACP.fit_transform(X)

    def generate_test(self,X):

        return self.ACP.transform(X)

    def generate_variables(self,Xtrain,Xtest):

        Xtrain = self.generate_train(Xtrain)
        Xtest = self.generate_test(Xtest)

        return (Xtrain,Xtest)

    def serialize(self):

        view = self.ACP.get_params()
        params = {k:v for k,v in view.items()}

        return Model.hash_model(params)