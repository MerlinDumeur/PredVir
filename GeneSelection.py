import Model


class GeneSelector:

    def __init__(self,model):

        self.model = model


class GeneSelector_GLM(GeneSelector):

    def __init__(self,model):

        GeneSelector.__init__(self,model)

    def select_genes(self,X,Y):

        self.model.fit(X,Y)

        if 'CV' in self.model.__class__.__name__:

            coef = self.model.best_estimator.coef_

        else:

            coef = self.model.coef_

        index_keep = np.greater(coef,np.zeros(len(coef)))

        return index_keep


class GeneSelectorFile:

    def from_CVFILE(cvfile,model):

        return GeneSelectorFile(cvfile.filename + Model.get_modelhash(model) + '.pkl')

    def __init__(self,filename):

        self.df = pd.read_pickle(filename)

    def select_genes(self,i):

        idx = self.df.loc[:,i]
        return self.df.loc[idx].index

    def generate_file(dataset,cvfile,geneselector):

        df = pd.DataFrame(index=dataset.X.columns)

        for i,Xtrain,Ytrain,Xtest,Ytest in enumerate(dataset.split(cvfile)):

            idkeep = geneselector.select_genes(Xtrain,Ytrain)
            df.loc[:,i] = 0
            df.loc[idkeep,i] = 1

        df.to_pickle(cvfile.filename + Model.get_modelhash(geneselector.model) + '.pkl')
