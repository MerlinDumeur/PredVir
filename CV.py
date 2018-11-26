from sklearn.model_selection import KFold,RepeatedKFold,StratifiedKFold
import Dataset
import Constants
import pandas as pd
import numpy as np


class CV:

    def __init__(self,cv,cv_args={}):

        self.cv = cv
        self.cv_args = cv_args

    def split(self,X):

        cv = self.cv(**self.cv_args)

        return cv.split(X)


type_dict = {

    'KFold':KFold,
    'RepeatedKFold':RepeatedKFold,
    'StratifiedKFold':StratifiedKFold

}


class CV2:

    def __init__(self,cv_type='KFold',cv_args={}):

        self.cv_type = cv_type
        self.cv_args = cv_args

    def split(self,X,**kwargs):

        if isinstance(self.cv_type,str):
            cv = type_dict[self.cv_type](**self.cv_args)
        else:
            cv = self.cv_type(**self.cv_args)

        return cv.split(X,**kwargs)


class CV_FILE:

    def __init__(self,filename):

        self.filename = filename
        self.df = pd.read_pickle(filename)

    def get_n_splits(self):

        return len(self.df.columns)

    def from_args(base,nmois,cv,cv_args):

        foldername = Dataset.Dataset.get_foldername(base,nmois)
        filename = CV_FILE.generate_filename_CV(cv,cv_args)
        return CV_FILE(foldername + filename + '.pkl')

    def split(self,X,y,end=None,**kwargs):

        if end is None or end > len(self.df.columns):

            end = len(self.df.columns)

        for i in range(end):

            idx_train = self.df.loc[self.df.loc[:,i] == 1].index
            idx_test = self.df.index.difference(self.df.loc[idx_train].index)

            yield (idx_train,idx_test)

    def generate_file(dataset,nmois,cv,cv_args,strata=None):

        cv_instance = cv(**cv_args)
        df = pd.DataFrame(index=dataset.X.index,columns=np.arange(cv_instance.get_n_splits()))

        for i,(train_index,test_index) in enumerate(cv_instance.split(dataset.X,strata)):

                df.loc[:,i] = 0
                df.iloc[train_index,i] = 1

        df.to_pickle(Dataset.Dataset.get_foldername(dataset.base,nmois) + CV_FILE.generate_filename_CV(cv,cv_args) + '.pkl')

    def generate_filename_CV(cv,cv_args):

        cv_instance = cv(**cv_args)

        if cv.__name__ in Constants.CV_DICT_KF.keys():

            argsvalues = CV_FILE.serialize_CV(cv_instance)
            argslist = [*argsvalues]
            argslist.sort()

            strlist = [f"{{{k}}}" for k in argslist]
            filename = f"{cv.__name__}" + '-' + "-".join(strlist).format(**argsvalues)

        else:

            argslist = cv.__code__.varnames.sort()
            filename = f"{cv.__class__.__name__}" + "".join([f"-{{{k}}}" for k in argslist]).format(**cv_args)

        return filename

    def serialize_CV(cv_instance):

        if cv_instance.__class__.__name__ in Constants.CV_DICT_KF:

            argslist = ['n_splits','random_state',*Constants.CV_DICT_KF[cv_instance.__class__.__name__]]
            argslist.sort()
            argsvalues = {'n_splits':cv_instance.get_n_splits(),
                          **{k:getattr(cv_instance,k) for k in ['random_state',*Constants.CV_DICT_KF[cv_instance.__class__.__name__]] if k != "n_splits"}}

            return argsvalues

        else:

            raise ValueError('CV pas dans la liste authorisee')
