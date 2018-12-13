# import Constants
# import Preprocessing
import Dataset
import CV
import GeneSelection
import ModelTester
import Model
import KernelLogisticRegression
# import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss,roc_auc_score, make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV, KFold,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import os
# from sklearn.metrics.pairwise import rbf_kernel

base = "epilung2"

# On traite les données brutes pour l'apprentissage

# Pr = Preprocessing.Preprocesser(base)

# rename_XpG = {"id_sample":Constants.ID,"age_min":Constants.AGE,"index_histology_code":Constants.HISTOLOGY,'os':Constants.OS,"id_pathology":Constants.PATHOLOGY,'efs':Constants.EFS,'dead':Constants.DEAD}
# keep_XpG = ['id_sample','sex','age_min','age_max','id_topology','id_topology_group','id_morphology','id_pathology','t','n','m','tnm_stage','dfs_months','os_months','relapsed','dead','treatment','exposure','index_histology_code','efs','os','lof_gof']
# dropna_XpG = [Constants.HISTOLOGY,Constants.PATHOLOGY]

# keep_Plt = ['ID','GB_ACC','Gene Symbol','ENTREZ_GENE_ID','RefSeq Transcript ID']
# rename_Plt = {'ID':Constants.ID}
# dtype_Plt = {'SPOT_ID':str}

# Pr.load_files(keep_XpG=keep_XpG,rename_XpG=rename_XpG,dropna_XpG=dropna_XpG,transpose_Trscr=True,keep_Plt=keep_Plt,rename_Plt=rename_Plt,dtype_Plt=dtype_Plt,remove_ctrl_Plt=True)

# nmlist = [None,6,12,60]

# Pr.generate_XY(nmonths_list=nmlist,standardize=False)

# On choisit les paramètres avec lesquels on travaille

seed = 45

nmois = 6
ds = Dataset.Dataset(base,nmois)

cv = KFold
cv_args = {'n_splits':5,'random_state':seed}

# geneSelector_metric = 'neg_log_loss'
geneSelector_metric = make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1])
geneSelector_cv = cv(**cv_args)

# modelscv_scoring = 'neg_log_loss'
modelscv_scoring = make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1])
models_cv = cv(**cv_args)

cv_primary = RepeatedStratifiedKFold
cv_primary_args = {'n_splits':5,'n_repeats':5}

# On cree le gene selector

LogisticRegression_Cdict = np.logspace(-4,1,20)
LR_grid = {'C':LogisticRegression_Cdict}

LogR1 = LogisticRegression(penalty='l1',solver='liblinear')
LogR1_cv = GridSearchCV(LogR1,LR_grid,scoring=geneSelector_metric,cv=geneSelector_cv,n_jobs=-1,iid=False)
LogR1_m = Model.Model(LogR1_cv,'LogR1')

GS = GeneSelection.GeneSelector_GLM(LogR1_m)

# print(zlib.compressobj(LogR1_cv))
# print(zlib.adler32(LogR1_cv))

# sys.exit()

# On genere les fichiers CV et Geneselction

# CV.CV_FILE.generate_file(ds,nmois,cv_primary,cv_primary_args,strata=ds.Y)
cvfile = CV.CV_FILE.from_args(base,nmois,cv_primary,cv_primary_args)

# GeneSelection.GeneSelectorFile.generate_file(ds,cvfile,GS)
gsfile = GeneSelection.GeneSelectorFile.from_CVFILE(cvfile,GS)


# On créée nos modèles

LR_grid = {'C':LogisticRegression_Cdict}

LR2 = LogisticRegression(solver='liblinear')
LR2_CV = Model.Model(GridSearchCV(LR2,LR_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=False),'LR2')
LR2_m = Model.Model(LR2_CV,'LogR2')

KLR_RBF_Gdict = np.logspace(-6,1,20)
KLR_RBF_Cdict = np.logspace(-1,1.2,10)
KLR_RBF_grid = {'C':LogisticRegression_Cdict,'gamma':KLR_RBF_Gdict}

KLR2_RBF = KernelLogisticRegression.KernelLogisticRegression(solver='liblinear',kernel='rbf')
KLR2_RBF_CV = Model.Model(GridSearchCV(KLR2_RBF,KLR_RBF_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=False),'KLR2_RBF')

KLR2_POL_Gdict = np.logspace(-6,-2.7,10)
KLR2_POL_Degdict = [2,3,4,5,6]
KLR2_POL_coef0dict = np.logspace(-5,2.2,15)
KLR_POL_grid = {'C':LogisticRegression_Cdict,'gamma':KLR2_POL_Gdict,'coef0':KLR2_POL_coef0dict,'degree':KLR2_POL_Degdict}

KLR2_POL = KernelLogisticRegression.KernelLogisticRegression(solver='lbfgs',kernel='polynomial',max_iter=5000)
KLR2_POL_CV = Model.Model(GridSearchCV(KLR2_POL,KLR_POL_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=False),'KLR2_POL')

KLR2_SIG_coef0dict = np.logspace(2.5,5.6,10)
KLR2_SIG_grid = {'C':LogisticRegression_Cdict,'coef0':KLR2_SIG_coef0dict}

KLR2_SIG = KernelLogisticRegression.KernelLogisticRegression(solver='liblinear',kernel='polynomial')
KLR2_SIG_CV = Model.Model(GridSearchCV(KLR2_SIG,KLR2_SIG_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=False),'KLR2_SIG')

models_list = [LR2_CV,KLR2_RBF_CV,KLR2_POL_CV,KLR2_SIG_CV]

# On teste les performances de nos modeles

perf_dict = {}
# metrics = {'log_loss':log_loss,'AUC':roc_auc_score}
metrics = {'log_loss':log_loss,'AUC':roc_auc_score,'accuracy':accuracy_score}
df_means = pd.DataFrame()

directory = f'{base}/{nmois}/{gsfile.hash}/'

if not os.path.exists(directory):
    os.makedirs(directory)

replace = True

for model in models_list:

    filename = directory + f'{model.name}.pkl'

    if not(os.path.exists(filename)) or replace:

        print(f"Currently testing {model.name}")
        MT = ModelTester.ModelTester(model)
        perf_dict[model.name] = MT.test_score(ds,cvfile,gsfile,metrics=metrics,save=rf'{model.name}-perf.pkl')
        perf_dict[model.name].to_pickle(filename)
        df_means = df_means.append(perf_dict[model.name].mean(),ignore_index=True)

# for n,df in perf_dict.items():

#     df.to_pickle(f'{base}/{nmois}/{gsfile.serialize()}/{n}.pkl')

df_means.to_pickle(f'{base}/{nmois}/{gsfile.hash}/means.pkl')

# with open('df.pkl','wb') as handle:
#     pickle.dump(perf_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
