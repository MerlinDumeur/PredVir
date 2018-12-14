import Constants

import pandas as pd
from matplotlib import pyplot as plt


class Resultats:

    def __init__(self,dataset):

        self.ds = dataset

    def generate_barplot(self,gs,modelsNames,metric,showStd=True):

        if isinstance(gs,int):
            directory = self.ds.get_foldername() + Constants.FOLDERPATH_GS.format(hash=gs)
        else:
            directory = self.ds.get_foldername() + Constants.FOLDERPATH_GS.format(gs.hash)

        df_means = pd.read_pickle(directory + 'means.pkl')
        df_vars = pd.read_pickle(directory + 'vars.pkl')

        MI = pd.MultiIndex.from_tuples(df_means.columns)

        df_means = df_means.reindex(MI,axis=1)
        df_vars = df_vars.reindex(MI,axis=1)

        L = df_means.loc[modelsNames,('Test',metric)]
        X = range(len(L))

        plt.bar(X,L)
        plt.ylabel(metric)
        plt.title(f'Gene selection #{gs if isinstance(gs,int) else gs.hash}')
        plt.xticks(X,L.keys())
        plt.show()
