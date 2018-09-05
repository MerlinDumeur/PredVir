import pandas as pd
from Constants import *
from sklearn import preprocessing


class Preprocesser:

    def __init__(self,base):

        self.base = base

    def load_files(keep_XpG=[],rename_XpG={},dropna_XPG=[],dtype_XpG={},transpose_Trscr=False,keep_Plt=[],rename_Plt={},dtype_Plt={},remove_ctrl_Plt=False):

        process_XpG(keep_XpG,rename_XpG,dropna_XPG,dtype_XpG)
        process_Trscr(transpose_Trscr)
        process_Plt(keep_Plt,rename_Plt,dtype_Plt,remove_ctrl_Plt)

    def feature_engineering(self,categorical_columns_XpG):

        self.XpG = OHEncoding(self.XpG,categorical_columns_XpG)

    def generate_XY(self,nmonths_list,standardize=False):

        if standardize:

            preprocessing.scale(self.XpG,copy=False)

        for n in [None, *nmonths_list]:

            index_n = Preprocesser.index_classification(self.XpG,n)

            X = self.XpG.loc[index_n]
            Y = create_Y(index_regression,n)

            name = 'regression' if n is None else f'classification-{nmois}'

            X.to_pickle(self.base + rf'/X_{name}.pkl')
            Y.to_pickle(base + f'Y_{name}.pkl')

    def process_XpG(self,keep,rename,dropna_col,dtype):

        self.XpG = pd.read_csv(self.base + XPG_FILEPATH,usecols=keep,dtypes=dtype)
        self.XpG.rename(index=str,columns=rename,inplace=True)
        self.XpG.dropna(axis=0,subset=dropna_col,inplace=True)
        self.XpG.set_index(ID,inplace=True)

    def process_Trscr(self,transpose):

        self.Trscr = pd.read_csv(base + DATA_FILEPATH)

        if transpose:

            self.Trscr = self.Trscr.transpose()

    def process_Plt(self,keep,rename,dtype,remove_control):

        self.Plt = pd.read_csv(base + PLT_FILEPATH)

        if remove_control:

            self.Plt = Plt.loc[Plt['SPOT_ID'] != '--Control']

        self.Plt = Plt.loc[:,keep]
        self.Plt.rename(index=str,columns=rename,inplace=True)

    def OHenc(df,categorical_columns):

        return pd.get_dummies(DF,columns=categorical_columns)

    def index_regression(df):
        
        return df.loc[:,DEAD].index.values

    def index_classification(df,n_mois):
    
        I = [(k[-1] == '+' and int(k[:-1]) < n_mois) for k in df[OS].values]
        return np.setdiff1d(df.index.values,I)

    def create_Y(self,index,n=None):

        if n is None:

            return self.XpG.loc[index,OS]

        else:

            return self.XpG.loc[index,OS].apply(f_survival(n))

    def f_survival(n):

        return lambda x: 1 if x[-1] == '+' else int((int(x) > n_mois))
