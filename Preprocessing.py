import pandas as pd
import Constants
from sklearn import preprocessing
import os
import numpy as np


class Preprocesser:

    def __init__(self,base):

        self.base = base

    def load_files(self,keep_XpG=[],rename_XpG={},dropna_XpG=[],dtype_XpG={},transpose_Trscr=False,keep_Plt=[],rename_Plt={},dtype_Plt={},remove_ctrl_Plt=False,use_plt=True):

        self.process_Trscr(transpose_Trscr)
        self.process_XpG(keep_XpG,rename_XpG,dropna_XpG,dtype_XpG)
        if use_plt:
            self.process_Plt(keep_Plt,rename_Plt,dtype_Plt,remove_ctrl_Plt)

    def feature_engineering(self,categorical_columns_XpG):

        self.XpG = Preprocesser.OHenc(self.XpG,categorical_columns_XpG)

    def generate_XY(self,nmonths_list,standardize=False,dataformat='epilung',var_type='OS',censor=True):

        if standardize:

            preprocessing.scale(self.XpG,copy=False)

        if censor:

            XpG_ref = self.XpG.copy()

            print(XpG_ref.shape)

        for n in nmonths_list:

            folder = Constants.FOLDERPATH.format(base=self.base,nmois=(f'{n}' if n is not None else 'R'))

            if not os.path.exists(folder):
                os.makedirs(folder)

            if censor:

                # drop_I = self.XpG[~self.XpG[Constant.DEAD] & self.XpG[Constants.OS] < n].index

                if dataformat == 'epilung':
                    # print(self.XpG[Constants.OS].unique())
                    alive = self.XpG.loc[self.XpG[Constants.OS].apply(lambda x: x[-1] == '+'),:]
                    alive[Constants.OS] = alive[Constants.OS].apply(lambda x: int(x[:-1]))
                    drop_I = alive[alive[Constants.OS] <= n].index

                else:
                    drop_I = self.XpG[(self.XpG[Constants.DEAD] == False) & (self.XpG[Constants.OS] <= n)].index
                self.XpG.drop(drop_I,inplace=True)
                # drop_I = self.XpG[(self.XpG[Constants.RELAPSED] == False) & (self.XpG[Constants.DEAD] == True) & (self.XpG[Constants.OS] < n)].index
                # self.XpG.drop(drop_I,inplace=True)

                if dataformat == 'epilung':
                    # alive = self.XpG.loc[self.XpG[Constants.OS].apply(lambda x: x[-1] == '+'),:].index
                    drop_I = alive[alive[Constants.OS] <= 100].index
                else:
                    drop_I = self.XpG[(self.XpG[Constants.DEAD] == False) & (self.XpG[Constants.OS] <= 100)].index
                self.XpG.loc[drop_I,:].to_pickle(folder + 'Undead.pkl')

            print(self.XpG.shape)

            index = Preprocesser.index(self.XpG,n,dataformat=dataformat)

            X = self.Trscr.loc[index]
            Y = self.create_Y(index,dataformat,n,var_type)


            X.to_pickle(folder + Constants.FILENAME_X)
            Y.to_pickle(folder + Constants.FILENAME_Y)

            if censor:

                self.XpG = XpG_ref.copy()

    def generate_OS(self,nmonths_list,use_dfs=True):

        for n in nmonths_list:

            folder = Constants.FOLDERPATH.format(base=self.base,nmois=(f'{n}' if n is not None else 'R'))

            OS = self.XpG[Constants.OS].apply(lambda x: int(x[:-1] if x[-1] == '+' else x))
            # OS = self.XpG[Constants.OS]
            OS = OS.apply(np.ceil)
            OS.to_pickle(folder + 'OS.pkl')

            if use_dfs:

                DFS = self.XpG[Constants.EFS]
                DFS = DFS.apply(np.ceil)
                DFS.to_pickle(folder + 'DFS.pkl')


    def generate_toyDataset(self,nmonths_list,size_data,size_var,standardize=False):

        if standardize:

            preprocessing.scale(self.XpG,copy=False)

        for n in nmonths_list:

            index = Preprocesser.index(self.XpG,n)

            X = self.Trscr.loc[index]
            Y = self.create_Y(index,n)

            X = X.iloc[1:size_data,1:size_var]
            Y = Y.iloc[1:size_data]

            folder = Constants.FOLDERPATH.format(base=self.base,nmois=(f'{n}' if n is not None else 'R'))

            if not os.path.exists(folder):
                os.makedirs(folder)

            X.to_pickle(folder + Constants.FILENAME_X)
            Y.to_pickle(folder + Constants.FILENAME_Y)

    def process_XpG(self,keep,rename,dropna_col,dtype):

        self.XpG = pd.read_csv(self.base + Constants.XPG_FILEPATH,usecols=keep,dtype=dtype,low_memory=False)
        self.XpG.rename(index=str,columns=rename,inplace=True)
        self.XpG.set_index(Constants.ID,inplace=True)
        self.XpG = self.XpG.loc[self.Trscr.index]
        self.XpG.dropna(axis=0,subset=dropna_col,inplace=True)

    def process_Trscr(self,transpose):

        self.Trscr = pd.read_csv(self.base + Constants.DATA_FILEPATH,low_memory=False,index_col=0)

        if transpose:

            self.Trscr = self.Trscr.transpose()

        self.Trscr = self.Trscr.dropna()

    def process_Plt(self,keep,rename,dtype,remove_control):

        self.Plt = pd.read_csv(self.base + Constants.PLT_FILEPATH,low_memory=False)

        if remove_control:

            self.Plt = self.Plt.loc[self.Plt['SPOT_ID'] != '--Control']

        self.Plt = self.Plt.loc[:,keep]
        self.Plt.rename(index=str,columns=rename,inplace=True)

    def OHenc(df,categorical_columns):

        return pd.get_dummies(Constants.DF,columns=categorical_columns)

    def index(df,n,**kwargs):

        if n is not None:

            return Preprocesser.index_classification(df,n,**kwargs)

        else:

            return Preprocesser.index_regression(df,**kwargs)

    def index_regression(df,**kwargs):

        return df.loc[:,Constants.DEAD].index.values

    def index_classification(df,n_mois,dataformat):
        
        if dataformat == 'epilung':
            Idx = [(k[-1] == '+' and int(k[:-1]) < n_mois) for k in df[Constants.OS].values]
        else:
            Idx = [not(d) and os < n_mois for d,os in df[[Constants.DEAD,Constants.OS]].values]
        
        return np.setdiff1d(df.index.values,Idx)

    def create_Y(self,index,dataformat,n=None,type='OS',censor=True):

        if n is None:

            return self.XpG.loc[index,Constants.OS]

        elif type == 'DFS':

            return self.XpG.loc[index,Constants.EFS].apply(Preprocesser.f_survival(n,dataformat))

        else:
            
            return self.XpG.loc[index,Constants.OS].apply(Preprocesser.f_survival(n,dataformat))

    def f_survival(n,dataformat):

        if dataformat == 'epilung':
        
            return lambda x: 1 if x[-1] == '+' else int((int(x) > n))

        else:
            
            return lambda x: int(x > n)
