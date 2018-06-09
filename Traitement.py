import numpy as np
import pandas as pd
import Model_selection as ms
from sklearn import preprocessing
import requests
import json
import time

HISTOLOGY = "histology"
SEX = "sex"
ID = "id"
AGE = "age"
OS = "Overall survival"
EFS = "Event Free Survival"
PATHOLOGY = "pathology"
DEAD = "dead"


def refining_expGrp(base,keep,rename,dropna_col):
    XpG = pd.read_csv(base + r'/exp_grp.csv')
    XpG = XpG[keep]
    XpG = XpG.rename(index=str,columns=rename)
    XpG = XpG.dropna(axis=0,subset=dropna_col)
    XpG.set_index(ID,inplace=True)
    return XpG


def refining_trscr(base,transpose=False):
    Trscr = pd.read_csv(base + r'/data.csv',index_col=0)
    if transpose:
        Trscr = Trscr.transpose()
    return Trscr


def refining_platform(base,keep,rename,dtype,remove_control=False):
    Plt = pd.read_csv(base + r'/platform.csv',dtype=dtype)
    if remove_control:
        Plt = Plt[Plt['SPOT_ID'] != '--Control']
    Plt = Plt[keep]
    Plt = Plt.rename(index=str,columns=rename)
    return Plt


def OHEncoding(filename,categorical_columns):
    DF = pd.read_pickle(filename)
    DF = pd.get_dummies(DF,columns=categorical_columns)
    return DF


def Create_Patient_Index_Regression(df):
    return df[DEAD].index.values


def Create_Patient_Drop_Index_Classification(df,n_mois):
    I = [(k[-1] == '+' and int(k[:-1]) < n_mois) for k in df[OS].values]
    return np.setdiff1d(df.index.values,I)


def X_transcriptome(base,index,standardize=True):
    X = pd.read_pickle(base + r'/trscr.pkl')
    if standardize:
        preprocessing.scale(X,copy=False)
    return X.loc[index]


def get_function_survival(n_mois):
    return lambda x: 1 if x[-1] == '+' else int((int(x) > n_mois))


def Y_clinique(base,index,n_mois=None):
    Y = pd.read_pickle(base + r'/clinique_OH.pkl')
    Y = Y.loc[index]
    if n_mois is None:
        Y = Y['os_months']
    else:
        Y = Y[OS].apply(get_function_survival(n_mois))
    return Y


def import_X(base,nmois=None,std=True):
    
    classifieur = nmois is not None
    nmois_str = f'-{nmois}' if classifieur else ""

    return pd.read_pickle(base + rf'/X_{"classification" + nmois_str if classifieur else "regression"}{"" if std else "_nostd"}.pkl')


def import_index_fs(base,id,cv_primary,classifieur_fr,classifieur_fs,nmois=None,std=True):

    filename = ms.get_filename(cv_primary,id,nmois,std)
    foldername = ms.get_foldername('FS',base,classifieur_fr,classifieur_fs)

    df = pd.read_pickle(foldername + filename)

    return df


def import_index_cv(base,id,cv_primary,nmois=None,std=True):

    filename = ms.get_filename(cv_primary,id,nmois,std)
    foldername = ms.get_foldername('CV_primary',base)

    df = pd.read_pickle(foldername + filename)

    return df


def import_Y(base,nmois=None):

    classifieur = nmois is not None
    nmois_str = f'-{nmois}' if classifieur else ""

    return pd.read_pickle(base + rf'/Y_{"classification" + nmois_str if classifieur else "regression"}.pkl')


def import_val_test_XY_cv_fs(base,id,cv_primary,cv_i,nmois=None,std=True,classifieur_fr=None,classifieur_fs=None):

    fs = (classifieur_fr is not None) and (classifieur_fs is not None)

    X = import_X(base=base,nmois=nmois,std=std)
    Y = import_Y(base=base,nmois=nmois)

    Index_cv = import_index_cv(base,id,cv_primary,nmois,std)

    IndexVal,IndexTest = ms.get_test_train_indexes(Index_cv,cv_i)

    Y_val = Y.loc[IndexVal]
    Y_test = Y.loc[IndexTest]

    if fs:
        Index_fs = import_index_fs(base,id,cv_primary,classifieur_fr,classifieur_fs,nmois=nmois,std=std)
        Indexfs = Index_fs.loc[Index_fs[cv_i]].index

        X_fs = X.loc[:,Indexfs]

    else:
        X_fs = X
    
    X_val = X_fs.loc[IndexVal]
    X_test = X_fs.loc[IndexTest]

    return X_fs,X_val,Y_val,X_test,Y_test


def update_genes(X,filename=None):

    array_s = X.columns.values
    l = array_s.tolist()
    dict_s = {s:s for s in l}

    for v in dict_s:
        if 'HS.' in dict_s[v]:
            dict_s[v] = 'Hs' + dict_s[v][2:]
        elif '.' in dict_s[v]:
            dict_s[v] = dict_s[v][:dict_s[v].rfind('.')]

    dict_s_inv = {v:k for k,v in dict_s.items()}

    chaine = ', '.join(str(e) for e in l)

    query_api_epimed(chaine)

    df = pd.read_csv('temp_geneupdate.csv',sep=';',index_col=0)
    df = df.dropna(axis=0,subset=['gene_symbol'])
    df = df.rename(index=dict_s_inv)

    X = X.loc[:,df.index]
    X = X.rename(columns=df['gene_symbol'])

    if filename is not None:
        X.to_pickle(filename)

    return X


def query_api_epimed(chaine):

    url = 'http://epimed.univ-grenoble-alpes.fr/database/query/'

    r = requests.get(url + 'jobid')

    json_l = json.loads(r.text)
    jobid = json_l['jobid']

    data = {"taxid":'9606','jobid':jobid,'symbols':chaine}

    requests.post(url + 'genes/update',data)

    finished = False

    while not finished:

        r3 = requests.get(url + 'jobstatus?jobid=' + jobid)
        json3 = json.loads(r3.text)

        finished = json3['status'] in ['succes','error']

        time.sleep(300)

    if json3['status'] == 'error':

        raise Exception('EPIMED ERROR')

    r4 = requests.get(url + 'jobs?jobid=' + jobid)
    
    # r4_d = r4.content.decode('utf-8')
    # csv_r = csv.reader(r4_d.splitlines(), delimiter=',')

    f = open('temp_geneupdate.csv','w')
    f.write(r4.text)
    f.close()


# taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also:
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    window_len = window_len + window_len % 2 - 1

    if x.ndim != 1:
        raise ValueError('smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(),s,mode='valid')

    margin = window_len // 2

    return y[margin:-margin]


"""
def barsplot(df,titre='',figsize=(6,4)):
    
    valeurs = df.unique()
    
    categories_sain = []
    categories_malade = []
    
    for v in valeurs:
        
        if pd.isnull(v):
            categories_sain.append(Clinique['id_pathology'].loc[Index_sain].apply(pd.isnull).sum())
            categories_malade.append(Clinique['id_pathology'].loc[Index_malade].apply(pd.isnull).sum())
        else:
            categories_malade.append((df.loc[Index_malade].values == v).sum())
            categories_sain.append((df.loc[Index_sain].values == v).sum())

    index = np.arange(valeurs.shape[0])
    
    fig=plt.figure(figsize=figsize)
    
    p1 = plt.bar(index,categories_malade)
    p2 = plt.bar(index,categories_sain,bottom=categories_malade)

    plt.ylabel('N')
    plt.title(titre)
    plt.xticks(index,valeurs)
    plt.legend((p2[0],p1[0]),('Sain','Malade'))
    plt.show()
"""
