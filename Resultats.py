import Constants

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Resultats:

    def __init__(self,dataset):

        self.ds = dataset

    def generate_barplot(self,gs,modelsNames,metric,showStd=True):

        if isinstance(gs,int) or isinstance(gs,str):
            directory = self.ds.get_foldername() + Constants.FOLDERPATH_GS.format(hash=gs)
        else:
            directory = self.ds.get_foldername() + Constants.FOLDERPATH_GS.format(gs.hash)

        df_means = pd.read_pickle(directory + 'means.pkl')
        df_vars = pd.read_pickle(directory + 'vars.pkl')

        MI = pd.MultiIndex.from_tuples(df_means.columns)

        df_means = df_means.reindex(MI,axis=1)
        df_vars = df_vars.reindex(MI,axis=1)

        L = df_means.loc[modelsNames,('Test',metric)]
        V = df_vars.loc[modelsNames,('Test',metric)]
        X = range(len(L))

        plt.bar(X,L,yerr=V)
        plt.ylabel(metric)
        plt.title(f'Gene selection #{gs if isinstance(gs,int) or isinstance(gs,str) else gs.hash}')
        plt.xticks(X,L.keys())
        plt.show()
        
    def generate_group_barplot(self,gs_list,modelsNames,metric,showStd=True,legend=[],save=None):
        
        X = None
        W = 0.35
        
        # p = []
        
        for i,gs in enumerate(gs_list):
        
            if isinstance(gs,int) or isinstance(gs,str):
                directory = self.ds.get_foldername() + Constants.FOLDERPATH_GS.format(hash=gs)
            else:
                directory = self.ds.get_foldername() + Constants.FOLDERPATH_GS.format(gs.hash)

            df_means = pd.read_pickle(directory + 'means.pkl')
            df_vars = pd.read_pickle(directory + 'vars.pkl')

            MI = pd.MultiIndex.from_tuples(df_means.columns)

            df_means = df_means.reindex(MI,axis=1)
            df_vars = df_vars.reindex(MI,axis=1)

            L = df_means.loc[modelsNames,('Test',metric)]
            V = df_vars.loc[modelsNames,('Test',metric)]
            
            X = X if X is not None else np.arange(len(L))

            plt.bar(X,L,W,yerr=V)
            plt.ylabel(metric)
#             plt.title(f'Gene selection #{gs if isinstance(gs,int) or isinstance(gs,str) else gs.hash}')
            X = X + W
        
        plt.xticks(np.arange(len(L))+W/(len(gs_list)),L.keys())
        plt.ylim([0.0,0.8])
        plt.legend(legend,loc='upper left')

        if save is not None:

            plt.savefig(save)

        else:
            
            plt.show()

    def barplot(perf_dict,metric,showStd=True):

        model_names = [*perf_dict]

        df_means = pd.DataFrame()
        df_vars = pd.DataFrame()

        for m in model_names:

            df_means = df_means.append(perf_dict[m].mean(),ignore_index=True)
            df_vars = df_vars.append(perf_dict[m].var(),ignore_index=True)

            # df_means.loc[m,:] = perf_dict[m].mean()
            # df_vars.loc[m,:] = perf_dict[m].var()

        rename = {i:n for i,n in enumerate(model_names)}
        df_means = df_means.rename(index=rename)
        df_vars = df_vars.rename(index=rename)

        MI = pd.MultiIndex.from_tuples(df_means.columns)

        df_means = df_means.reindex(MI,axis=1)
        df_vars = df_vars.reindex(MI,axis=1)

        L = df_means.loc[model_names,('Test',metric)]
        X = range(len(L))

        plt.bar(X,L)
        plt.ylabel(metric)
        # plt.title(f'Gene selection #{gs if isinstance(gs,int) else gs.hash}')
        plt.xticks(X,L.keys())
        plt.show()