from sklearn.model_selection import KFold,RepeatedKFold,StratifiedKFold


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

    def __init__(self,seed,n_splits,cv_type=KFold,cv_args={}):

        self.seed = seed
        self.n_splits = n_splits
        self.cv_type=cv_type
        self.cv_args=cv_args

    def split(self,X,**kwargs):

        cv = type_dict[self.cv_type](n_splits=self.n_splits,random_state=self.seed,**self.cv_args)

        return cv.split(X,**kwargs)

class CV_FILE:

    def __init__(self,filename):

        self.df = pd.read_pickle(filename)

    def split(self,end=None,**kwargs):

        if end is None or end > len(self.df.index):

            end = len(self.df.index)

        for i in range(end):

            yield self.df.loc[:,i]

    def generate_file(dataset,nmois,CV,cv_args):

        X = 'Regression' if n is None else f'{n}nmois'