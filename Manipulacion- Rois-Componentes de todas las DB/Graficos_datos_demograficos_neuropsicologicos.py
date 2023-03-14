import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD'

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

data_Comp=pd.read_feather(r"{path}\Datosparaorganizardataframes\Data_complete_roi.feather".format(path=path))#Datos con sujetos sin datos vaciosbands= data_Comp['Band'].unique()graphic_dem(data_Comp,'group','age',None,'swarm',0.5,None,'Distribucion de edades por cada grupo en cada base de datos','swarm',save=True,plot=False)
data_Comp = data_Comp[(data_Comp['visit'] == 'V0') | (data_Comp['visit'] == 't1')]
graphic_dem(data_Comp,'group','age',None,'box',0.5,None,path,'Distribucion de edades por cada grupo en cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp,'sex','age','group','swarm',0.5,None,path,'Distribucion de edades por genero en cada grupo de cada base de datos','swarm',save=True,plot=False)
graphic_dem(data_Comp,'sex','age','group','box',0.5,None,path,'Distribucion de edades por genero en cada grupo de cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['BIOMARCADORES','DUQUE','CHBMP'])],'group','education',None,'swarm',0.5,0.075,path,'Distribucion de años de escolaridad por cada grupo en cada base de datos','swarm',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['BIOMARCADORES','DUQUE','CHBMP'])],'group','education',None,'box',0.5,0.075,path,'Distribucion de años de escolaridad por cada grupo en cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['BIOMARCADORES','DUQUE','CHBMP'])],'group','MM_total',None,'box',0.5,0.075,path,'Distribucion de MM_total por cada grupo en cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['BIOMARCADORES','DUQUE','CHBMP'])],'group','MM_total',None,'swarm',0.5,0.075,path,'Distribucion de MM_total por cada grupo en cada base de datos','swarm',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','BIOMARCADORES'])],'group','FAS_F',None,'box',0.5,0.075,path,'Distribucion de FAS_F por cada grupo en cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','BIOMARCADORES'])],'group','FAS_F',None,'swarm',0.5,0.075,path,'Distribucion de FAS_F por cada grupo en cada base de datos','swarm',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','BIOMARCADORES'])],'group','FAS_S',None,'box',0.5,0.075,path,'Distribucion de FAS_S por cada grupo en cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','BIOMARCADORES'])],'group','FAS_S',None,'swarm',0.5,0.075,path,'Distribucion de FAS_S por cada grupo en cada base de datos','swarm',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','BIOMARCADORES'])],'group','FAS_A',None,'box',0.5,0.075,path,'Distribucion de FAS_A por cada grupo en cada base de datos','box',save=True,plot=False)
graphic_dem(data_Comp[data_Comp.database.isin(['SRM','BIOMARCADORES'])],'group','FAS_A',None,'swarm',0.5,0.075,path,'Distribucion de FAS_A por cada grupo en cada base de datos','swarm',save=True,plot=False)

print('Graficos de datos demograficos guardados')
