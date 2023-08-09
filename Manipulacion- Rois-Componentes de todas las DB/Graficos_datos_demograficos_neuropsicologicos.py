import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Funciones import estadisticos_demograficos

path_save=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados'
path = rf'{path_save}/dataframes'

neuro = 'sovaHarmony'
name = 'G1'
space = 'ic'
var = ''
def graphic_dem(data,x_data,y_data,col,kind,bbox_x,left,path,title,id,save=True,plot=True):
    
    """Function to make graphs of demographic data"""

    sns.set(rc={'figure.figsize':(15,12)})
    sns.set_theme(style="white")
    if kind=='box':
        linewidth=0.5
    else:
        linewidth=None
    if left==None:
        left=0.05
    axs=sns.catplot(x=x_data,y=y_data,data=data,hue='database',col=col,dodge=True,palette='winter_r', kind=kind,linewidth=linewidth,legend=False)
    axs.fig.suptitle(title)
    axs.add_legend(loc='upper center',bbox_to_anchor=(bbox_x,.94),ncol=4,title="Database")
    axs.fig.subplots_adjust(top=0.78,bottom=0.121, right=0.986,left=left, hspace=0.138, wspace=0.062) # adjust the Figure in rp
    if plot==True:
        plt.show()
    if save==True:
        plt.savefig('{path}\Graficos_datos_demograficos_neuropisicologicos\{title}-{id}.png'.format(path=path,title=title,id=id))
        plt.close()
    
    return 
## Paths
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
tk.Tk().withdraw() # part of the import if you are not using other tkinter functions
path = askdirectory()
print("user chose", path, "for read feather")
path_feather = path + r'\dataframes'
os.makedirs(path_feather,exist_ok=True)
path_input = path + r'\dataframes\Data_complete_'
os.makedirs(path_input,exist_ok=True)
#data_Comp=pd.read_feather(rf'{path_save}/dataframes/{neuro}/integration/{space}/{name}/Data_complete_{space}_{neuro}_{name}.feather')#Datos con sujetos sin datos vaciosbands= data_Comp['Band'].unique()graphic_dem(data_Comp,'group','age',None,'swarm',0.5,None,'Distribucion de edades por cada grupo en cada base de datos','swarm',save=True,plot=False)
##### PARA ARTICULO #####
data_Comp= pd.read_feather(path_input+space+'.feather') #Para articulo
data_Comp = data_Comp[(data_Comp['visit'] == 'V0') | (data_Comp['visit'] == 't1')] #Para articulo
data_Comp.loc[data_Comp['group']=='G2','group'] = 'Control' #Para articulo
data_Comp = data_Comp[(data_Comp['group'] == 'G1') | (data_Comp['group'] == 'Control')] #Para articulo
##########################

#data_Comp.loc[data_Comp['database']=='BIOMARCADORES','database'] = 'Cohort 1'
#data_Comp.loc[data_Comp['database']=='DUQUE','database'] = 'Cohort 2'
#data_Comp.loc[data_Comp['database']=='CHBMP','database'] = 'Cohort 3'
#data_Comp.loc[data_Comp['database']=='SRM','database'] = 'Cohort 4'
#path_std = rf'{path_save}\tables\{neuro}\{space}\{name}'
path_std = rf'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados\tables\{neuro}\{space}\{name}'
estadisticos_demograficos(data_Comp,'ic',path_std)
graphic_dem(data_Comp,'group','age',None,'box',0.5,None,path,'Age Distribution by Group in Each Database','box',save=False,plot=True)
graphic_dem(data_Comp,'sex','age','group','swarm',0.5,None,path,'Age Distribution by Gender in Each Group of Each Database','swarm',save=False,plot=True)
graphic_dem(data_Comp,'sex','age','group','box',0.5,None,path,'Age Distribution by Gender in Each Group of Each Database','box',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['Cohort 1','Cohort 2','CHBMP'])],'group','education',None,'swarm',0.5,0.075,path,'Distribution of Years of Education by Each Group in Each Database','swarm',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['Cohort 1','Cohort 2','CHBMP'])],'group','education',None,'box',0.5,0.075,path,'Distribution of Years of Education by Each Group in Each Database','box',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['Cohort 1','Cohort 2','CHBMP'])],'group','MM_total',None,'box',0.5,0.075,path,'Distribution of MM_total by Each Group in Each Database','box',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['Cohort 1','Cohort 2','CHBMP'])],'group','MM_total',None,'swarm',0.5,0.075,path,'Distribution of MM_total by Each Group in Each Database','swarm',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','Cohort 1'])],'group','FAS_F',None,'box',0.5,0.075,path,'Distribution of FAS_F by Each Group in Each Database','box',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','Cohort 1'])],'group','FAS_F',None,'swarm',0.5,0.075,path,'Distribution of FAS_F by Each Group in Each Database','swarm',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','Cohort 1'])],'group','FAS_S',None,'box',0.5,0.075,path,'Distribution of FAS_S by Each Group in Each Database','box',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','Cohort 1'])],'group','FAS_S',None,'swarm',0.5,0.075,path,'Distribution of FAS_S by Each Group in Each Database','swarm',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','Cohort 1'])],'group','FAS_A',None,'box',0.5,0.075,path,'Distribution of FAS_A by Each Group in Each Database','box',save=False,plot=True)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','Cohort 1'])],'group','FAS_A',None,'swarm',0.5,0.075,path,'Distribution of FAS_A by Each Group in Each Database','swarm',save=False,plot=True)
print('Graficos de datos demograficos guardados')
