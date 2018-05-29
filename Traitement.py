import numpy as np
import pandas as pd
from sklearn import preprocessing

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

    return pd.read_pickle(base + rf'/X_{"classification" + nmois_str if classifieur else "regression"}_{"" if std else "nostd"}.pkl')


def import_Y(base,predicteur_type,nmois=None):

    classifieur = nmois is not None
    nmois_str = f'-{nmois}' if classifieur else ""

    return pd.read_pickle(base + rf'/Y_{"classification" + nmois_str if classifieur else "regression"}.pkl')


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
