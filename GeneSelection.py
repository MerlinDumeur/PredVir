import pandas as pd
import numpy as np


class GeneSelector:

    def __init__(self,model):

        self.model = model


class GeneSelector_GLM(GeneSelector):

    def __init__(self,model):

        GeneSelector.__init__(self,model)

    def select_genes(self,X,Y):

        # print(X)
        # print(Y)

        self.model.fit(X,Y)

        if 'CV' in self.model.model.__class__.__name__:

            coef = self.model.best_estimator_.coef_

        else:

            coef = self.model.coef_

        index_keep = np.greater(coef,np.zeros(len(coef)))

        return index_keep[0]

    def serialize(self):

        return self.model.get_modelhash()


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

    def generate_file(dataset,cvfile,geneselector):

        df = pd.DataFrame(index=dataset.X.columns)

        for i,(Xtrain,Ytrain,Xtest,Ytest) in enumerate(dataset.CV_split(cvfile)):

            idkeep = geneselector.select_genes(Xtrain,Ytrain)
            df.loc[:,i] = 0
            df.loc[idkeep,i] = 1

        df.to_pickle(cvfile.filename[:-4] + str(geneselector.model.get_modelhash()) + '.pkl')
