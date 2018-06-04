from sklearn.metrics import log_loss,make_scorer
from sklearn.model_selection import RepeatedKFold,KFold,GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression,Lasso,Ridge
from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

import Traitement as proc
import Predicteurs as cl
import Model_selection as ms
import pandas as pd
import pickle
import os

import matplotlib.pyploy as plt
import numpy as np


def visualize_model(base,nmois,id,cv_primary,classifieur_fr,classifieur_fs,cv_i,taille_maille,**kwargs):

    std = kwargs.get('std',True)

    Index_fs = proc.import_index_fs(base,id,cv_primary,classifieur_fr,classifieur_fs,nmois=nmois,std=std)
    Index_cv = proc.import_index_cv(base,id,cv_primary,nmois,std)

    X = proc.import_X(base=base,nmois=nmois,std=std)
    Y = proc.import_Y(base,nmois)

    IndexVal,IndexTest = ms.get_test_train_indexes(Index_cv,cv_i)
    Indexfs = Index_fs.loc[Index_fs[cv_i]].index

    X_fs = X.loc[:,Indexfs]

    ACP = PCA(n_components=2)

    ACP.fit(X_fs)

    X_val = X_fs.loc[IndexVal]
    Y_val = Y.loc[IndexVal]

    X_test = X_fs.loc[IndexTest]
    Y_test = Y.loc[IndexTest]

    X_valr = ACP.transform(X_val)
    X_testr = ACP.transform(X_test)

    x_min = min(X_valr[:,0].min(), X_testr[:,0].min()) - .5
    x_max = max(X_valr[:,0].max(), X_testr[:,0].max()) + .5
    
    y_min = min(X_valr[:,1].min(), X_testr[:,1].min()) - .5
    y_max = max(X_valr[:,1].max(), X_testr[:,1].max()) + .5

    xx, yy = np.meshgrid(np.arange(x_min,x_max,taille_maille),np.arange(y_min,y_max,taille_maille))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    ax = plt.plot()
    ax.scatter(X_valr[:,0],X_valr[:,1])