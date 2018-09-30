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

    def __init__(self,seed,n_splits):

        self.seed = seed
        self.n_splits = n_splits

    def split(self,X,cv_type='KFold',cv_args={},**kwargs):

        cv = type_dict[cv_type](n_splits=self.n_splits,random_state=self.seed)

        return cv.split(X,**kwargs)
