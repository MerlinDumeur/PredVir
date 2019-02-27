# # import Constants
# import Preprocessing
import Dataset
import CV
import GeneSelection
# import ModelTester
import Model
import KernelLogisticRegression
from ModelTester import _flatten_grid
import Constants
# # import AdaptativeSearchCV
# # import SequentialGridSearchCV
# # import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss,roc_auc_score, make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV, KFold,RepeatedStratifiedKFold,RepeatedKFold
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
import os

seed = 45
nmois = 30

base1 = 'ADSC'
base2 = 'nsclc'

ds1 = Dataset.Dataset(base1,nmois)
ds2 = Dataset.Dataset(base2,nmois)

cv = KFold
cv_args = {'n_splits':5,'random_state':seed}

geneSelector_metric = make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1])
geneSelector_cv = cv(**cv_args)

modelscv_scoring = make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1])
models_cv = cv(**cv_args)

LogisticRegression_Cdict = np.logspace(-4,1,20)
LR_grid = {'C':LogisticRegression_Cdict}

LogR1 = LogisticRegression(penalty='l1',solver='liblinear')
LogR1_cv = GridSearchCV(LogR1,LR_grid,scoring=geneSelector_metric,cv=geneSelector_cv,n_jobs=-1,iid=False)
LogR1_cv = GridSearchCV(LogR1,{'C':[0.5]},scoring=geneSelector_metric,cv=geneSelector_cv,n_jobs=-1,iid=False)
LogR1_m = Model.Model(LogR1_cv,'LogR1')
LogR1GS_iterative = GeneSelection.GeneSelector_GLM_Iterative(LogR1_m,700)

KLR_RBF_Gdict = np.logspace(-6,-3,15)
KLR_RBF_Cdict = np.logspace(-1,1,10)
KLR_RBF_grid = {'C':KLR_RBF_Cdict,'gamma':KLR_RBF_Gdict}

KLR2_RBF = KernelLogisticRegression.KernelLogisticRegression(solver='liblinear',kernel='rbf')
KLR2_RBF_CV = Model.Model(GridSearchCV(KLR2_RBF,KLR_RBF_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=True),'KLR2_RBF')

geneSelector = LogR1GS_iterative
model = KLR2_RBF_CV

metrics = {'log_loss':log_loss,'AUC':roc_auc_score,'accuracy':accuracy_score}

Xtrain,Ytrain = ds1.X,ds1.Y
Xtest,Ytest = ds2.X,ds2.Y

idx_commun = Xtrain.columns.intersection(Xtest.columns)

Xtrain = Xtrain.loc[:,idx_commun]
Xtest = Xtest.loc[:,idx_commun]

MI = pd.MultiIndex.from_arrays([['general'] * 2,['fs_size','Class']])

arrays_m = [['Test'] * (len(metrics) + 1),['score',*metrics]]
MI2 = pd.MultiIndex.from_arrays(arrays_m)

param_grid = model.param_grid
# print(*_flatten_grid(param_grid))
flattened_grid = _flatten_grid(param_grid)
arrays_p = [['Validation'] * (len(flattened_grid) + 1),['score',*flattened_grid]]
MI3 = pd.MultiIndex.from_arrays(arrays_p)

MI_final = MI.copy()
MI_final = MI_final.append(MI2)
MI_final = MI_final.append(MI3)

df_output = pd.DataFrame(index=[0],columns=MI_final)

idx_genes = geneSelector.select_genes(Xtrain,Ytrain)

Xtrain = Xtrain.loc[:,idx_genes]
Xtest = Xtest.loc[:,idx_genes]

model.fit(Xtrain,Ytrain)

best_params = model.best_params_
score = model.score(Xtest,Ytest)

i = 0

df_output.loc[i,'general'] = Xtrain.shape[1]
df_output.loc[i,'Class'] = model.best_estimator_.__class__.__name__

if len(metrics) == 0:
    df_output.loc[i,'Test'] = score
else:
    # print(Ytest.shape)
    # print(model.predict_proba(Xtest).shape)
    # print(metrics['accuracy'](Ytest,model.predict(Xtest)))
    df_output.loc[i,'Test'] = [score] + [metrics[m](Ytest,model.predict_proba(Xtest)[:,1]) if metrics[m].__name__ not in Constants.no_proba_list else metrics[m](Ytest,model.predict(Xtest)) for m in arrays_m[1][1:]]

# print(best_params)

if len(best_params) == 0:
    df_output.loc[i,'Validation'] = model.best_score_
else:
    df_output.loc[i,'Validation'] = [model.best_score_] + [best_params[k] for k in arrays_p[1][1:]]

df_pred = pd.Series(data=model.predict_proba(Xtest)[:,1],index=Xtest.index)

directory = ds2.get_foldername() + Constants.FOLDERPATH_GS.format(hash=geneSelector.serialize())

if not os.path.exists(directory):
    os.makedirs(directory)

df_output.to_pickle(directory + model.name + '.pkl')
df_pred.to_pickle(directory + 'pred_' + model.name + '.pkl')
