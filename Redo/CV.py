from sklearn.model_selection import KFold,RepeatedKFold,StratifiedKFold
import Dataset


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

    def from_args(base,nmois,CV,cv_args):

        foldername = Dataset.get_foldername(base,nmois)
        filename = CV_FILE.generate_filename_CV(CV,cv_args)
        return CV_FILE(foldername + filename + '.pkl')

    def split(self,end=None,**kwargs):

        if end is None or end > len(self.df.index):

            end = len(self.df.index)

        for i in range(end):

            idx_train = self.df.loc[:,i]
            idx_test = self.df.index.difference(self.df.loc[idx_train].index)

            yield (idx_train,idx_test)

    def generate_file(dataset,nmois,CV,cv_args,strata=None):

        cv = CV(**cv_args)
        df = pd.DataFrame(index=dataset.X.index,columns=np.arange(cv.get_n_splits()))
        
        cvs = enumerate(cv.split(dataset.X,strata)) if strata is not None else enumerate(cv.split(dataset.X))

        for i,train_index,test_index in cvs:

                df.loc[:,i] = 0
                df.loc[train_index,i] = 1

        df.to_pickle(CV_FILE.generate_filename_CV(CV,cv_args) + '.pkl')

    def generate_filename_CV(CV,cv_args):

        argslist = CV.__code__.varnames.sort()
        filename = f"{CV.__name__}" + str.join([f"-{{{k}}}" for k in argslist]).format(**cv_args)
        return filename
