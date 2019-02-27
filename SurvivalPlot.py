import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class SurvivalPlot:
    """docstring for SurvivalPlot"""

    def __init__(self):

        pass

    def load_data(self,ds,gs,model_name,n_repeat_cv):

        filename = ds.get_foldername() + f'{gs}/pred_{model_name}.pkl'
        self.df_pred = pd.read_pickle(filename)

        self.folds = []
        print(self.df_pred)
        size_fold = int(np.floor(self.df_pred.shape[1] / n_repeat_cv))

        for k in range(n_repeat_cv):

            self.folds.append([self.df_pred[i + k * size_fold].dropna() for i in range(size_fold)])

        self.preds_fold = [pd.concat(f) for f in self.folds]

        self.undead = pd.read_pickle(ds.get_foldername() + 'Undead.pkl')

        self.OS = pd.read_pickle(ds.get_foldername() + 'OS.pkl')
        self.DFS = pd.read_pickle(ds.get_foldername() + 'DFS.pkl')

        self.OS = self.OS[self.preds_fold[0].index].dropna().drop(self.undead.index)
        self.DFS = self.DFS[self.preds_fold[0].index].dropna().drop(self.undead.index)

    def load_data_unique(self,ds,gs,model_name):

        filename = ds.get_foldername() + f'{gs}/pred_{model_name}.pkl'
        self.df_pred = pd.read_pickle(filename)

        self.preds_fold = [self.df_pred]

        self.undead = pd.read_pickle(ds.get_foldername() + 'Undead.pkl')

        self.OS = pd.read_pickle(ds.get_foldername() + 'OS.pkl')
        # self.DFS = pd.read_pickle(ds.get_foldername() + 'DFS.pkl')

        self.OS = self.OS[self.preds_fold[0].index].dropna().drop(self.undead.index)
        # self.DFS = self.DFS[self.preds_fold[0].index].dropna().drop(self.undead.index)

    def plot_distribution(self,fold_id):

        plt.plot(np.sort(self.preds_fold[fold_id]))

    def plot_survival(self,fold_id,thresholds,variable='OS',grid=np.linspace(1,100,100),save=None):

        # Computing index of each group

        thresholds = np.sort(thresholds)

        self.gr = {}

        if fold_id == "mean":

            preds_fold = pd.DataFrame(data=self.preds_fold).transpose().mean(axis=1)

        else:

            preds_fold = self.preds_fold[fold_id]
        
        preds_fold = preds_fold[self.OS.index]

        # print(len(self.df_pred[fold_id]))
        # print(max(self.df_pred[fold_id].values))

        for i in range(len(thresholds)):
            
            I = preds_fold[preds_fold < thresholds[i]].index
            # print(len(I))
            diff = I
            
            for idx in self.gr.values():
                
                diff = diff.difference(idx)

            self.gr[i] = diff

        # Computing surviving patients for each month in self.grid for each self.group

        groups = {k: [len(g)] for k,g in self.gr.items()}

        for k,g in self.gr.items():

            print(f'Group {k}: {len(g)}')

        indexes = self.gr

        Export = pd.Series(index=self.preds_fold[0].index)

        for i in range(len(indexes)):

            Export[indexes[i]] = i + 1

        Export.to_pickle('groups.pkl')

        for g in grid:
            
            if variable == 'OS':
                I = self.OS[self.OS < g].index
            else:
                I = self.DFS[self.DFS < g].index
            
            for k in groups:
                groups[k].append(groups[k][0] - len(I.intersection(indexes[k])))

        # Normalisation

        for k in groups:
            
            groups[k] = [s / groups[k][0] for s in groups[k]]

        # Plotting

        for k in groups:
            
            plt.plot(groups[k],drawstyle='steps-post')
            
        plt.legend([len(g) for g in self.gr.values()])
        plt.xlabel('Months')
        plt.ylabel('Percentage of patients alive')
        plt.ylim((0,1))

        if save is not None:

            plt.savefig(save)

        else:
            
            plt.show()
