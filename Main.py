# import Constants
# import Preprocessing
import Dataset
import CV
import GeneSelection
import ModelTester
import Model
import KernelLogisticRegression
import Constants
# import AdaptativeSearchCV
import SequentialGridSearchCV
# import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss,roc_auc_score, make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV, KFold,RepeatedStratifiedKFold,RepeatedKFold
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

nmois = 60
ds = Dataset.Dataset(base,nmois)

cv = KFold
cv_args = {'n_splits':5,'random_state':seed}

# geneSelector_metric = 'neg_log_loss'
geneSelector_metric = make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1])
geneSelector_cv = cv(**cv_args)

# modelscv_scoring = 'neg_log_loss'
modelscv_scoring = make_scorer(log_loss,greater_is_better=False,needs_proba=True,labels=[0,1])
models_cv = cv(**cv_args)

# cv_primary = RepeatedKFold
# cv_primary_args = {'n_splits':5,'n_repeats':5}

cv_primary = KFold
cv_primary_args = {'n_splits':5,'random_state':seed}

# On cree le gene selector

LogisticRegression_Cdict = np.logspace(-4,1,20)
LR_grid = {'C':LogisticRegression_Cdict}

LogR1 = LogisticRegression(penalty='l1',solver='liblinear')
LogR1_cv = GridSearchCV(LogR1,LR_grid,scoring=geneSelector_metric,cv=geneSelector_cv,n_jobs=-1,iid=False)
LogR1_m = Model.Model(LogR1_cv,'LogR1')

LogR2 = LogisticRegression(solver='liblinear')
LogR2_CV = GridSearchCV(LogR2,LR_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=True)
LogR2_m = Model.Model(LogR2_CV,'LogR2')

LogR1GS = GeneSelection.GeneSelector_GLM(LogR1_m)

PGS = GeneSelection.GeneSelector_Participative(model_fr=LogR1_m,model_fs=LogR2_m,cv_fr=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=seed),cv_fs=geneSelector_cv,metric_fs=log_loss)

# print(zlib.compressobj(LogR1_cv))
# print(zlib.adler32(LogR1_cv))

# sys.exit()

# On genere les fichiers CV et Geneselction

# CV.CV_FILE.generate_file(ds,nmois,cv_primary,cv_primary_args,strata=ds.Y)
cvfile = CV.CV_FILE.from_args(base,nmois,cv_primary,cv_primary_args)


# GeneSelection.GeneSelectorFile.generate_file(ds,cvfile,PGS,weight_name='coef_',grid=np.linspace(0.1,0.5,20),savename='AVG_file.pkl')
print("Generated PGS file")

# GeneSelection.GeneSelectorFile.generate_file(ds,cvfile,LogR1GS)
print("Generated LogR1GS file")

PGS = GeneSelection.GeneSelectorFile.from_CVFILE(cvfile,PGS)
GS = GeneSelection.GeneSelectorFile.from_CVFILE(cvfile,LogR1GS)
GS_ps = GeneSelection.GeneSelectorFile.from_CVFILE(cvfile,LogR1GS,True)

# On créée nos modèles

LR_grid = {'C':LogisticRegression_Cdict}

LR2 = LogisticRegression(solver='liblinear')
LR2_CV = Model.Model(GridSearchCV(LR2,LR_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=True),'LR2')
# LR2_m = Model.Model(LR2_CV,'LogR2')

LR1 = LogisticRegression(penalty='l1',solver='liblinear')
LR1_cv = Model.Model(GridSearchCV(LogR1,LR_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=False),'LR1')

KLR_RBF_Gdict = np.logspace(-6,1,20)
KLR_RBF_Cdict = np.logspace(-1,1.2,10)
KLR_RBF_grid = {'C':LogisticRegression_Cdict,'gamma':KLR_RBF_Gdict}

KLR2_RBF = KernelLogisticRegression.KernelLogisticRegression(solver='liblinear',kernel='rbf')
KLR2_RBF_CV = Model.Model(GridSearchCV(KLR2_RBF,KLR_RBF_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=True),'KLR2_RBF')

KLR2_POL_Gdict = np.logspace(-6,-2.7,10)
KLR2_POL_Degdict = [2,3,4,5,6]
KLR2_POL_coef0dict = np.logspace(-5,2.2,15)
KLR_POL_grid = {'C':LogisticRegression_Cdict,'gamma':KLR2_POL_Gdict,'coef0':KLR2_POL_coef0dict,'degree':KLR2_POL_Degdict}

KLR2_POL = KernelLogisticRegression.KernelLogisticRegression(solver='liblinear',kernel='polynomial',max_iter=100)
KLR2_POL_CV = Model.Model(GridSearchCV(KLR2_POL,KLR_POL_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=True),'KLR2_POL')

KLR2_SIG_coef0dict = np.logspace(2.5,5.6,10)
KLR2_SIG_grid = {'C':LogisticRegression_Cdict,'coef0':KLR2_SIG_coef0dict}

KLR2_SIG = KernelLogisticRegression.KernelLogisticRegression(solver='liblinear',kernel='sigmoid')
KLR2_SIG_CV = Model.Model(GridSearchCV(KLR2_SIG,KLR2_SIG_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1,iid=True),'KLR2_SIG')

XGB_maxdepth = [1,2,3,4]
XGB_lr = np.logspace(-1.5,-0.5,5)
XGB_nestimators = [10,100,200]
XGB_alpha = np.logspace(-9,-6,5)
XGB_beta = np.logspace(-9,-6,5)

XGB_grid1 = {"alpha":XGB_alpha,"beta":XGB_beta}
XGB_grid2 = {"max_depth":XGB_maxdepth,"learning_rate":XGB_lr,"n_estimators":XGB_nestimators}
XGB_seq_grid = [XGB_grid1,XGB_grid2]
XGB_grid = dict(XGB_grid1, **XGB_grid2)

XGB = XGBClassifier(n_jobs=4)
# XGB_CV = Model.Model(SequentialGridSearchCV.SequentialGridSearchCV(XGB,XGB_seq_grid,scoring='neg_log_loss',cv=models_cv,n_jobs=1,iid=True),'XGB')
XGB_CV = Model.Model(GridSearchCV(XGB,XGB_grid,scoring='neg_log_loss',cv=models_cv,n_jobs=1,iid=True),'XGB')

XGB2_CV = Model.Model(GridSearchCV(XGBClassifier(n_jobs=4),{},scoring=modelscv_scoring,cv=models_cv,n_jobs=1,iid=True),'XGB_noparam')

# gs_list = []
gs_list = [PGS,GS_ps,GS]
models_list = [LR2_CV,KLR2_RBF_CV,KLR2_SIG_CV,XGB_CV,XGB2_CV]


# On teste les performances de nos modeles

perf_dict = {}
# metrics = {'log_loss':log_loss,'AUC':roc_auc_score}
metrics = {'log_loss':log_loss,'AUC':roc_auc_score,'accuracy':accuracy_score}

for gsfile in gs_list:

    print(f'Starting gsfile')

    directory = ds.get_foldername() + Constants.FOLDERPATH_GS.format(hash=gsfile.hash)

    filename_means = directory + 'means.pkl'
    filename_vars = directory + 'vars.pkl'

    df_means = pd.DataFrame() if not(os.path.exists(filename_means)) else pd.read_pickle(filename_means)
    df_vars = pd.DataFrame() if not(os.path.exists(filename_vars)) else pd.read_pickle(filename_vars)

    if not os.path.exists(directory):
        os.makedirs(directory)

    replace = True
    resume = False

    for model in models_list:

        filename = directory + f'{model.name}.pkl'

        if not(os.path.exists(filename)) or replace:

            print(f"Currently testing {model.name}")
            MT = ModelTester.ModelTester(model)
            perf_dict[model.name] = MT.test_score(ds,cvfile,gsfile,metrics=metrics,save=filename,resume=resume)
            # perf_dict[model.name].to_pickle(filename)
            df_means.loc[model.name,:] = perf_dict[model.name].mean()
            df_vars.loc[model.name,:] = perf_dict[model.name].var()

    # for n,df in perf_dict.items():

    #     df.to_pickle(f'{base}/{nmois}/{gsfile.serialize()}/{n}.pkl')

    df_means.to_pickle(filename_means)
    df_vars.to_pickle(filename_vars)

# with open('df.pkl','wb') as handle:
#     pickle.dump(perf_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
