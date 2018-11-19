import Constants
import Preprocessing
import Dataset
import CV
import GeneSelection
import ModelTester
import Model

import numpy as np

from sklearn.metrics import log_loss,roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

base = "epilung2"

# On traite les données brutes pour l'apprentissage

Pr = Preprocessing.Preprocesser(base)

# keep_XpG = ['id_sample','sex','age_min','age_max','id_topology','id_topology_group','id_morphology','id_pathology','t','n','m','tnm_stage','dfs_months','os_months','relapsed','dead','treatment','exposure','index_histology_code','efs','os','lof_gof']
# rename_XpG = {"id_sample":Constants.ID,"age_min":Constants.AGE,"index_histology_code":Constants.HISTOLOGY,'os':Constants.OS,"id_pathology":Constants.PATHOLOGY,'efs':Constants.EFS,'dead':Constants.DEAD}
dropna_XpG = [Constants.HISTOLOGY,Constants.PATHOLOGY]

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

LogisticRegression_Cdict = np.logspace(-3,3,6)
LR_grid = {'C':LogisticRegression_Cdict}

LogR1 = LogisticRegression(penalty='l1')
LogR1_cv = GridSearchCV(LogR1,LR_grid,scoring=geneSelector_metric,cv=geneSelector_cv)

GS = GeneSelection.GeneSelector_GLM(LogR1_cv)

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

LR2 = LogisticRegression()
LR2_CV = Model.Model(GridSearchCV(LR2,LR_grid,scoring=modelscv_scoring,cv=models_cv,n_jobs=-1),'LR2')

models_list = [LR2_CV]

# On teste les performances de nos modeles

perf_dict = {}
metrics = {'log_loss':log_loss,'AUC':roc_auc_score}

for model in models_list:

    MT = ModelTester.ModelTester(model.model)
    perf_dict[model.name] = MT.test_score(ds,cvfile,gsfile,metrics=metrics,save=rf'{model.name}-perf.pkl')
