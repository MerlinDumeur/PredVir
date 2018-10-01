class GeneSelector:

    def __init__(self,model):

        self.model = model


class GeneSelector_GLM(GeneSelector):

    def __init__(self,model):

        GeneSelector.__init__(self,model)

    def select_genes(self,X,Y):

        self.model.fit(X,Y)
        coef = self.model.coef_
        index_keep = np.greater(coef,np.zeros(len(coef)))

        return index_keep

class GeneSelectorFile:

    def __init__(self,filename):

        self.df = pd.read_pickle(filename)

    def select_genes(self,i):

        return self.df.loc[:,i]