HISTOLOGY = "histology"
SEX = "sex"
ID = "id"
AGE = "age"
OS = "Overall survival"
EFS = "Event Free Survival"
PATHOLOGY = "pathology"
DEAD = "dead"

XPG_FILEPATH = '/exp_grp.csv'
DATA_FILEPATH = '/data.csv'
PLT_FILEPATH = '/platform.csv'

FOLDERPATH = '{base}/{nmois}/'
FILENAME_X = 'X.pkl'
FILENAME_Y = 'Y.pkl'

CV_DICT_KF = {

    'KFold':['shuffle'],
    'RepeatedKFold':['n_repeats'],
    'StratifiedKFold':['shuffle'],
    'RepeatedStratifiedKFold':['n_repeats']

}
