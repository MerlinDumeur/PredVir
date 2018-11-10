import numpy as np
import pandas as pd
import Model_selection as ms
from sklearn import preprocessing
import requests
import json
import time
from scipy.stats import iqr

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


def Get_Index(df,nmois=None):

    if nmois is not None:
        return Create_Patient_Index_Regression(df)
    
    return Create_Patient_Drop_Index_Classification(df,nmois)


def Get_Index2(baseList,nmois=None,std=True):

    IndexList = [import_X(b,nmois,std).index for b in baseList]
    Indexoutput = IndexList[0].append(IndexList[1:])

    return Indexoutput


# def X_transcriptome2(base,i,index,standardize=True,normalize=True):

#     X = pd.read_pickle(base + rf'/trscr_{i}'.pkl)

#     if standardize:
#         preprocessing.scale(X,copy=False)

#     return X.loc[index]


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


def generate_XY(base,nmois=None,std=True):

    Clinique_OH = pd.read_pickle(base + r'/clinique_OH.pkl')

    Index = Get_Index(Clinique_OH,nmois)

    X = X_transcriptome(base,Index,std)
    Y = Y_clinique(base,Index)

    name = 'regression' if nmois is None else f'classification-{nmois}'

    X.to_pickle(base + f'X_{name}.pkl')
    Y.to_pickle(base + f'Y_{name}.pkl')


def generate_XY2(baseDestination,baseList,nmois=None,std=True):

    Index = Get_Index2(baseList,nmois=None,std=True)

    X = X_transcriptome(baseDestination,Index,std)
    Ylist = [import_Y(b,nmois) for b in baseList]
    Y = Ylist[0].append(Ylist[1:])

    name = 'regression' if nmois is None else f'classification-{nmois}'

    X.to_pickle(baseDestination + f'X_{name}.pkl')
    Y.to_pickle(baseDestination + f'Y_{name}.pkl')


def add_bases_trscr(baseDestination,baseList,updateList=[],normalize=True):

    for b in updateList:

        T = pd.read_pickle(b + '/trscr.pkl')
        T = update_genes(T,b + f'/trscr_updated.pkl')

    genesCommon = get_common_index(baseList)

    Tlist = [pd.read_pickle(b + f'/trscr_updated.pkl') for b in baseList]

    if normalize:

        QT = preprocessing.QuantileTransformer(output_distribution='normal')

        for i,T in enumerate(Tlist):
    
            QT.fit(T)
            Tlist[i] = QT.transform(T)

    # Toutput = Tlist[0]

    # for T in Tlist[1:]:

    #     Toutput = Toutput.concat(T,axis=0,join_axes=[])

    Tcommon = pd.concat(Tlist,axis=0,join_axes=[genesCommon])

    Tcommon.to_pickle(baseDestination + '/trscr.pkl')


def get_common_index(baseList):

    T = pd.read_pickle(baseList[0] + '/trscr_updated.pkl')
    genesCommon = T.columns

    for b in baseList[1:]:

        T = pd.read_pickle(b + '/trscr_updated.pkl')
        genesCommon = genesCommon.intersection(T.columns)

    return genesCommon


def select_common_genes(X1,X2):

    genesCommon = X1.columns.intersection(X2.columns)
    X1c = X1.loc[X1.index,genesCommon]
    X2c = X2.loc[X2.index,genesCommon]

    return X1c,X2c


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

    f = 'epimed_query.csv'
    query_api_epimed(chaine,f)

    df = pd.read_csv(f,sep=';',index_col=0)
    df = df.dropna(axis=0,subset=['gene_symbol'])
    df = df.rename(index=dict_s_inv)

    Index_duplicate = df[df.loc[:,'gene_symbol'].duplicated(keep=False)].sort_values('gene_symbol',axis=0).index
    S_duplicate = df.loc[Index_duplicate,'gene_symbol'].sort_values()

    Index_split = df[df.index.duplicated(keep=False)].index
    Index_split = Index_split.drop_duplicates(keep='first')

    I_keep = remove_duplicates(S_duplicate,X)
    
    Index_normal = df.index.difference(Index_duplicate).difference(Index_split)
    Index_output = Index_normal.append(I_keep)

    X_output = X.loc[X.index,Index_output]

    print(X_output._is_view)
    X_output = split_genes(df.loc[Index_split,'gene_symbol'],X,X_output)

    for i in df.loc[Index_split,'gene_symbol']:

        if i not in X_output.columns:

            print('DansGame')

    X_output = X_output.rename(columns=df.loc[Index_output,'gene_symbol'])

    if filename is not None:
        X_output.to_pickle(filename)

    return X_output


def split_genes(df,X_input,X_output):

    I = df.index.values

    for i in range(len(I)):

        if df.iloc[i] not in X_output.columns:

            X_output.loc[:,df.iloc[i]] = X_input.loc[:,I[i]]

    return X_output


def remove_duplicates(S_dupes,X):

    i = 0
    liste = [i]
    last = S_dupes.values[0]
    for v in S_dupes.values[1:]:
        if v == last:
            liste.append(i)
        else:
            i += 1
            last = v
            liste.append(i)

    df = pd.DataFrame(columns=['gene_symbol','strata'],index=S_dupes.index)
    df['gene_symbol'] = S_dupes

    Strata = pd.Series(liste,index=S_dupes.index)
    df['strata'] = Strata

    array_keep = []

    for i,s in enumerate(df['strata'].unique()):

        a_genes = df[df['strata'] == s].index.values
        array_IQR = np.zeros(len(a_genes))

        for j,g in enumerate(a_genes):

            array_IQR[j] = iqr(X[g])

        array_keep.append(a_genes[np.argmax(array_IQR)])

    array_keep = np.array(array_keep)

    return pd.Index(array_keep)


def query_api_epimed(chaine,filename):

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

        finished = json3['status'] in ['success','error']

        time.sleep(300)

    if json3['status'] == 'error':

        raise Exception('EPIMED ERROR')

    r4 = requests.get(url + 'jobs?jobid=' + jobid)
    
    # r4_d = r4.content.decode('utf-8')
    # csv_r = csv.reader(r4_d.splitlines(), delimiter=',')

    f = open(filename,'w')
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
