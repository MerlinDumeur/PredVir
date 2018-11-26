import pandas as pd
import Constants
from sklearn import preprocessing
import os
import numpy as np


class Preprocesser:

    def __init__(self,base):

        self.base = base

    def load_files(self,keep_XpG=[],rename_XpG={},dropna_XpG=[],dtype_XpG={},transpose_Trscr=False,keep_Plt=[],rename_Plt={},dtype_Plt={},remove_ctrl_Plt=False):

        self.process_XpG(keep_XpG,rename_XpG,dropna_XpG,dtype_XpG)
        self.process_Trscr(transpose_Trscr)
        self.process_Plt(keep_Plt,rename_Plt,dtype_Plt,remove_ctrl_Plt)

    def feature_engineering(self,categorical_columns_XpG):

        self.XpG = Preprocesser.OHenc(self.XpG,categorical_columns_XpG)

    def generate_XY(self,nmonths_list,standardize=False):

        if standardize:

            preprocessing.scale(self.XpG,copy=False)

        for n in nmonths_list:

            index = Preprocesser.index(self.XpG,n)

            X = self.Trscr.loc[index]
            Y = self.create_Y(index,n)

            folder = Constants.FOLDERPATH.format(base=self.base,nmois=(f'{n}' if n is not None else 'R'))

            if not os.path.exists(folder):
                os.makedirs(folder)

            X.to_pickle(folder + Constants.FILENAME_X)
            Y.to_pickle(folder + Constants.FILENAME_Y)

    def process_XpG(self,keep,rename,dropna_col,dtype):

        self.XpG = pd.read_csv(self.base + Constants.XPG_FILEPATH,usecols=keep,dtype=dtype,low_memory=False)
        self.XpG.rename(index=str,columns=rename,inplace=True)
        self.XpG.dropna(axis=0,subset=dropna_col,inplace=True)
        self.XpG.set_index(Constants.ID,inplace=True)

    def process_Trscr(self,transpose):

        self.Trscr = pd.read_csv(self.base + Constants.DATA_FILEPATH,low_memory=False,index_col=0)

        if transpose:

            self.Trscr = self.Trscr.transpose()

    def process_Plt(self,keep,rename,dtype,remove_control):

        self.Plt = pd.read_csv(self.base + Constants.PLT_FILEPATH,low_memory=False)

        if remove_control:

            self.Plt = self.Plt.loc[self.Plt['SPOT_ID'] != '--Control']

        self.Plt = self.Plt.loc[:,keep]
        self.Plt.rename(index=str,columns=rename,inplace=True)

    def OHenc(df,categorical_columns):

        return pd.get_dummies(Constants.DF,columns=categorical_columns)

    def index(df,n):

        if n is not None:

            return Preprocesser.index_classification(df,n)

        else:

            return Preprocesser.index_regression(df)

    def index_regression(df):

        return df.loc[:,Constants.DEAD].index.values

    def index_classification(df,n_mois):

        Idx = [(k[-1] == '+' and int(k[:-1]) < n_mois) for k in df[Constants.OS].values]
        return np.setdiff1d(df.index.values,Idx)

    def create_Y(self,index,n=None):

        if n is None:

            return self.XpG.loc[index,Constants.OS]

        else:

            return self.XpG.loc[index,Constants.OS].apply(Preprocesser.f_survival(n))

    def f_survival(n):

        return lambda x: 1 if x[-1] == '+' else int((int(x) > n))
