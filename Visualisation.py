from sklearn.decomposition import PCA

from matplotlib.colors import ListedColormap

import Traitement as proc
import pandas as pd

import matplotlib.pyploy as plt
import numpy as np


class Visualizer:

    def __init__(self,taille_maille,coef=0,**kwargs):

        self.taille_maille = taille_maille
        self.coef = coef
        self.cm = kwargs.get('cm',plt.cm.RdBu)
        self.cm_bright = kwargs.get('cm_bright', ListedColormap(['#FF0000','#0000FF']))
        self.marker_size = kwargs.get('marker_size',None)
        self.figsize = kwargs.get('figsize',None)

    def display_base(self,n_dimensions,base,nmois,id,cv_primary,cv_i,classifieur_fr=None,classifieur_fs=None,std=True):

        X_fs,X_val,Y_val,X_test,Y_test = proc.import_val_test_XY_cv_fs(base,id,cv_primary,cv_i,nmois=nmois,std=std,classifieur_fr=classifieur_fr,classifieur_fs=classifieur_fs)

        self.display_XY(X_fs,X_val,Y_val,X_test,Y_test)

    def display_XY(self,n_dimensions,X_fs,X_val,Y_val,X_test,Y_test,classifieur_model):

        ACP = PCA(n_components=n_dimensions)
        ACP.fit(X_fs)

        X_valr = ACP.tranform(X_val)
        X_testr = ACP.transform(X_test)

        index_min = [min(X_valr[:,i].min(),X_testr[:,i].min()) - .5 for i in range(n_dimensions)]
        index_max = [max(X_valr[:,i].max(),X_testr[:,i].max()) + .5 for i in range(n_dimensions)]

        plt.figure(figsize=self.figsize)

        for i in range(n_dimensions - 1):
            for j in range(i + 1,n_dimensions):

                ax = plt.subplot(n_dimensions - 1,n_dimensions - 1,i + 1 + ((n_dimensions - 1) * (j - 1)))
                self.plotij(n_dimensions,i,j,ax,X_fs,X_val,Y_val,X_test,Y_test,ACP,index_min,index_max,classifieur_model)

        plt.tight_layout()
        plt.show()

    def display_ij(self,n_dimensions,X_fs,X_val,Y_val,X_test,Y_test,i,j,classifieur_model):

        ACP = PCA(n_components=n_dimensions)
        ACP.fit(X_fs)

        X_valr = ACP.tranform(X_val)
        X_testr = ACP.transform(X_test)

        index_min = [min(X_valr[:,i].min(),X_testr[:,i].min()) - .5 for i in range(n_dimensions)]
        index_max = [max(X_valr[:,i].max(),X_testr[:,i].max()) + .5 for i in range(n_dimensions)]

        plt.figure(figsize=self.figsize)

        ax = plt.subplot(1,1,1)
        self.plotij(n_dimensions,i,j,ax,X_fs,X_val,X_valr,Y_val,X_test,X_testr,Y_test,ACP,index_min,index_max,classifieur_model)

        plt.show()

    def plotij(self,n,i,j,axe,X_fs,X_val,X_valr,Y_val,X_test,X_testr,Y_test,ACP,IndexMin,IndexMax,classifieur_model):

        grid = np.meshgrid(*[[self.coef] if k not in [i,j] else np.arange(IndexMin[k],IndexMax[k],self.taille_maille) for k in range(n)])
            
        a = np.c_[tuple(g.ravel() for g in grid)]
        b = ACP.inverse_transform(a)
        df = pd.DataFrame(b,columns=X_val.columns)

        classifieur_model.fit(X_val,Y_val)
        Z = classifieur_model.predict_proba(df)[:,1]
        Z = Z.reshape(grid[0].shape)

        shape = grid[0].shape
        s = tuple(slice(None) if shape[k] != 1 else 0 for k in range(n))
            
        axe.contourf(grid[i][s],grid[j][s],Z[s],cmap=self.cm,alpha=.8)
            
        axe.scatter(X_valr[:,i],X_valr[:,j],c=Y_val,cmap=self.cm_bright,edgecolors='k',s=self.marker_size)
        axe.scatter(X_testr[:,i],X_testr[:,j],c=Y_test,cmap=self.cm_bright,edgecolors='k',alpha=0.6,s=self.marker_size)
        
        axe.set_xlim(grid[i].min(),grid[i].max())
        axe.set_ylim(grid[j].min(),grid[j].max())
            
        axe.set_xticks(())
        axe.set_yticks(())


class VisualizerBase(Visualizer):

    def __init__(self,base,nmois,id,cv_primary,cv_i,taille_maille,classifieur_fr=None,classifieur_fs=None,coef=0,**kwargs):

        cm = kwargs.get('cm',plt.cm.RdBu)
        cm_bright = kwargs.get('cm_bright', ListedColormap(['#FF0000','#0000FF']))
        marker_size = kwargs.get('marker_size',None)
        figsize = kwargs.get('figsize',None)

        Visualizer.__init__(taille_maille,coef,cm=cm,cm_bright=cm_bright,marker_size=marker_size,figsize=figsize)

        std = kwargs.get('std',True)
        
        self.X_fs,self.X_val,self.Y_val,self.X_test,self.Y_test = proc.import_val_test_XY_cv_fs(base,id,cv_primary,cv_i,nmois=nmois,std=std,classifieur_fr=classifieur_fr,classifieur_fs=classifieur_fs)

    def display_XY(self,n_dimension,classifieur_model):

        super().display_XY(n_dimension,self.X_fs,self.X_val,self.Y_val,self.X_test,self.Y_test,classifieur_model)

    def display_ij(self,n_dimension,i,j,classifieur_model):

        super().display_ij(n_dimension,self.X_fs,self.X_val,self.Y_val,self.X_test,self.Y_test,i,j,classifieur_model)
