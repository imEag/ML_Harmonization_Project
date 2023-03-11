
import collections
import pandas as pd 
import seaborn as sns
import numpy as np
import pingouin as pg
from numpy import ceil 
from matplotlib import pyplot as plt
import os
import io
from itertools import combinations
from PIL import Image
import matplotlib.pyplot as plt
import dataframe_image as dfi
from openpyxl import load_workbook
import warnings
warnings.filterwarnings("ignore")

def graphics_bar(data,type,path,name_band,id,id_cross=None,num_columns=None,save=True,plot=True):
    '''Function to make graphs of the given data '''
    sns.set(rc={'figure.figsize':(15,12)})
    sns.set_theme(style="white")
    # if id=='IC':
    #     col='Component'
    # else:
    #     col='ROI'
    axs=sns.catplot(x=id,y='effect size',data=data,hue='Compared groups',dodge=True, kind="bar",col=name_band,col_wrap=num_columns,linewidth=0.5,palette="cubehelix",legend=False)
    axs.set(xlabel=None)
    axs.set(ylabel=None)
    if id_cross==None:
        axs.fig.suptitle('Effect size on the metric '+r'$\bf{'+type+r'}$'+' in the '+r'$\bf{'+id+r'}$'+' of normalized data given by the databases',x=0.5,y=0.95)
    else:
        axs.fig.suptitle(r'$\bf{'+type+r'}$'+' in '+r'$\bf{'+id_cross+r'}$'+' of in the '+r'$\bf{'+id+r'}$'+' of normalized data given by the databases',x=0.5,y=0.95)


    if id=='Component':
        
        axs.add_legend(loc='upper center',ncol=6,title="Compared Groups",bbox_to_anchor=(0.5, 0.9))
        axs.fig.subplots_adjust(top=0.70,bottom=0.121, right=0.986,left=0.06, hspace=0.138, wspace=0.062) 
        axs.fig.text(0.5, 0.04, 'Group', ha='right', va='center')
        axs.fig.text(0.01, 0.5,  type, ha='right', va='center',rotation='vertical')
    else:
        
        axs.add_legend(loc='upper center',ncol=6,title="Compared Groups",bbox_to_anchor=(0.5, 0.9))
        axs.fig.subplots_adjust(top=0.70,bottom=0.121, right=0.986,left=0.06, hspace=0.138, wspace=0.062) # adjust the Figure in rp
        axs.fig.text(0.5, 0.04, 'Group', ha='right', va='center')
        axs.fig.text(0.015, 0.5,  type, ha='right', va='center',rotation='vertical')
    if plot:
        plt.show()
    if save==True:
        if id_cross==None:
            path_complete='{path}\Graficos_resultados_effect_size_sin_cv/bands_{type}_{id}.png'.format(path=path,id=id,type=type)  
        else:
            path_complete='{path}\Graficos_resultados_effect_size_sin_cv/mbands_{id_cross}_{type}_{id}.png'.format(path=path,id=id,type=type,id_cross=id_cross)
        plt.savefig(path_complete)
    plt.close()
    return 

path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' #Cambia dependieron de quien lo corra

#data loading

for i in ['Component','ROI']:
        
    data_cv=pd.read_excel("{path}\check_sin_cv.xlsx".format(path=path),sheet_name=i)
    data_cv=data_cv.sort_values(['metric',i,'Compared groups','band','mband'])
    metrics=data_cv['metric'].unique()
    bands=data_cv['band'].unique()
    bandsm= data_cv['mband'].unique()
    data=data_cv[data_cv['group']=='Control']
    for metric in metrics:
        print(metric)
        data_m=data[data['metric']==metric]
        if metric!='Cross Frequency':
            graphics_bar(data_m,metric,path,'band',i,id_cross=None,num_columns=4,save=True,plot=False)
            print('Done!')
        else:
            for band in bands:
                data_b=data_m[data_m['band']==band]
                if data_b.empty!=True:
                    graphics_bar(data_b,metric,path,'mband',i,id_cross=band,num_columns=4,save=True,plot=False)
                    print(band)
                    print('Done!')


