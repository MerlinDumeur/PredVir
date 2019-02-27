import pandas as pd
import Constants


class Dataset:

    def __init__(self,base,nmois):

        self.base = base
        self.nmois = nmois
        self.X,self.Y = self.load_XY()

    def CV_split(self,CV,strata=None):

        end = None

        for train_index,test_index in CV.split(self.X,strata,end=end):

            Xtrain = self.X.loc[train_index]
            Ytrain = self.Y.loc[train_index]

            Xtest = self.X.loc[test_index]
            Ytest = self.Y.loc[test_index]

            yield(Xtrain,Ytrain,Xtest,Ytest)

    def load_XY(self):

        self.folder = Dataset.foldername(self.base,self.nmois)
        X = pd.read_pickle(self.folder + Constants.FILENAME_X)
        Y = pd.read_pickle(self.folder + Constants.FILENAME_Y)

        return X,Y

    def get_foldername(self):

        return Dataset.foldername(self.base,self.nmois)

    def foldername(base,nmois):

        return Constants.FOLDERPATH.format(base=base,nmois=str(nmois) if nmois is not None else 'R')


class Dataset_XY:

    def __init__(self,X,Y):

        self.X = X
        self.Y = Y

    def CV_split(self,CV,strata=None):

        end = None

        for train_index,test_index in CV.split(self.X,strata,end=end):

            Xtrain = self.X.loc[train_index]
            Ytrain = self.Y.loc[train_index]

            Xtest = self.X.loc[test_index]
            Ytest = self.Y.loc[test_index]

            yield(Xtrain,Ytrain,Xtest,Ytest)
