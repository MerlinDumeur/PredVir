import pandas as pd
import Constants


class Dataset:

    def __init__(self,base,nmois):

        self.base = base
        self.X,self.Y = self.load_XY(base,nmois)

    def CV_split(self,CV,strata=None):

        for train_index,test_index in CV.split(self.X,strata):

            Xtrain = self.X.loc[train_index]
            Ytrain = self.Y.loc[train_index]

            Xtest = self.X.loc[test_index]
            Ytest = self.Y.loc[test_index]

            yield(Xtrain,Ytrain,Xtest,Ytest)

    def load_XY(self,base,nmois):

        self.folder = Dataset.get_foldername(base,nmois)
        X = pd.read_pickle(self.folder + Constants.FILENAME_X)
        Y = pd.read_pickle(self.folder + Constants.FILENAME_Y)

        return X,Y

    def get_foldername(base,nmois):

        return base + Constants.FOLDERPATH.format(base=base,nmois=str(nmois) if nmois is not None else 'R')
