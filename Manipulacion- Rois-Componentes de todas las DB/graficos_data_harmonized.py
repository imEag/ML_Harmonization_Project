import collections
import pandas as pd 
import seaborn as sns
import numpy as np
import pingouin as pg
from numpy import ceil 
import errno
from matplotlib import pyplot as plt
import os
import io
from itertools import combinations
from PIL import Image
import matplotlib.pyplot as plt
import dataframe_image as dfi
import warnings
from Funciones import dataframe_long_roi,dataframe_long_components,dataframe_componentes_deseadas,dataframe_long_cross_ic,dataframe_long_cross_roi
from Funciones import columns_SL_roi,columns_coherence_roi,columns_entropy_rois,columns_powers_rois
#from Graficos_power_sl_coherencia_entropia_cross import graphics
warnings.filterwarnings("ignore")

path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' #Cambia dependieron de quien lo corra

#data loading
data_roi_sova=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_All(SRM_CHBMP_G1_G2).feather'.format(path=path))
data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_All(SRM_CHBMP_G1_G2).feather'.format(path=path))

#Colimbas de cross frequency
DUQUE_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_crossfreq_columns_ROI_DUQUE.feather'.format(path=path))
columns_cross_roi=DUQUE_cr.columns.tolist()
for i in ['participant_id', 'group', 'visit', 'condition','database']:
    columns_cross_roi.remove(i)

for i,value in enumerate([data_roi_sova,data_roi_harmo]):
    d_B_roi=value
    if i==0:
        label="_sova"
    else:
        label="_harmonized"
    #New dataframes from ROIs
    #Dataframes are saved by ROI and components for graphics.
    #Power
    dataframe_long_roi(d_B_roi,'Power',columns=columns_powers_rois,name="data_long_power_roi_without_oitliers{label}".format(label=label),path=path)
    #SL
    dataframe_long_roi(d_B_roi,type='SL',columns=columns_SL_roi,name="data_long_sl_roi{label}".format(label=label),path=path)
    #Coherencia
    dataframe_long_roi(d_B_roi,type='Coherence',columns=columns_coherence_roi,name="data_long_coherence_roi{label}".format(label=label),path=path)
    #Entropia
    dataframe_long_roi(d_B_roi,type='Entropy',columns=columns_entropy_rois,name="data_long_entropy_roi{label}".format(label=label),path=path)
    #Cross frequency
    dataframe_long_cross_roi(d_B_roi,type='Cross Frequency',columns=columns_cross_roi,name="data_long_crossfreq_roi{label}".format(label=label),path=path)

def graphics(data,type,path,name_band,id,id2,id_cross=None,num_columns=4,save=True,plot=True):
    '''Function to make graphs of the given data '''
    max=data[type].max()
    sns.set(rc={'figure.figsize':(15,12)})
    sns.set_theme(style="white")
    if id=='IC':
        col='Component'
    else:
        col='ROI'
    axs=sns.catplot(x='group',y=type,data=data,dodge=True, kind="box",col=col,col_wrap=num_columns,palette='winter_r',fliersize=1.5,linewidth=0.5,legend=False)
    #plt.yticks(np.arange(0,round(max),0.1))
    axs.set(xlabel=None)
    axs.set(ylabel=None)
    if id_cross==None:
        axs.fig.suptitle(type+' in '+r'$\bf{'+name_band+r'}$'+ ' in the ICs of normalized data given by the databases')

        
    else:
        axs.fig.suptitle(type+' in '+id_cross+' of ' +r'$\bf{'+name_band+r'}$'+ ' in the ICs of normalized data given by the databases')
    if id=='IC':
        #axs.add_legend(loc='upper right',bbox_to_anchor=(.59,.95),ncol=4,title="Database")
        axs.fig.subplots_adjust(top=0.85,bottom=0.121, right=0.986,left=0.05, hspace=0.138, wspace=0.062) 
        axs.fig.text(0.5, 0.04, 'Group', ha='center', va='center')
        axs.fig.text(0.01, 0.5,  type, ha='center', va='center',rotation='vertical')
        
    else:
        #axs.add_legend(loc='upper right',bbox_to_anchor=(.7,.95),ncol=4,title="Database")
        axs.fig.subplots_adjust(top=0.85,bottom=0.121, right=0.986,left=0.06, hspace=0.138, wspace=0.062) # adjust the Figure in rp
        axs.fig.text(0.5, 0.04, 'Group', ha='center', va='center')
        axs.fig.text(0.015, 0.5,  type, ha='center', va='center',rotation='vertical')
        
    
    if plot:
        plt.show()
    if save==True:
        if id_cross==None:
            path_complete='{path}\Graficos_armonizacion_sova_harmo\Graficos_{type}\{id}\{name_band}_{type}_{id}_{id2}.png'.format(path=path,name_band=name_band,id=id,id2=id2,type=type)  
        else:
            path_complete='{path}\Graficos_armonizacion_sova_harmo\Graficos_{type}\{id}\{name_band}_{id_cross}_{type}_{id}_{id2}.png'.format(path=path,name_band=name_band,id=id,id2=id2,type=type,id_cross=id_cross)
        plt.savefig(path_complete)
    plt.close()
    return path_complete

labels=['_sova','_harmonized']
for label in labels:
    data_p_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_power_roi_without_oitliers{label}.feather'.format(label=label,path=path))
    data_sl_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_sl_roi{label}.feather'.format(label=label,path=path))
    data_c_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_coherence_roi{label}.feather'.format(label=label,path=path))
    data_e_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_entropy_roi{label}.feather'.format(label=label,path=path))
    data_cr_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_crossfreq_roi{label}.feather'.format(label=label,path=path))
    
    datos_roi={'Power':data_p_roi,'SL':data_sl_roi,'Coherence':data_c_roi,'Entropy':data_e_roi,'Cross Frequency':data_cr_roi}

    bands= data_sl_roi['Band'].unique()
    bandsm= data_cr_roi['M_Band'].unique()

    for metric in datos_roi.keys():
        for band in bands:
            d_roi=datos_roi[metric]
            d_banda_roi=d_roi[d_roi['Band']==band]
            if metric!='Cross Frequency':  
                print(str(band)+' '+str(metric)) 
                path_roi=graphics(d_banda_roi,metric,path,band,'ROI',label,num_columns=2,save=True,plot=False)
                
            else:
                for bandm in bandsm:  
                    print(str(band)+' '+str(metric)+' '+str(bandm)) 
                    if d_banda_roi[d_banda_roi['M_Band']==bandm]['Cross Frequency'].iloc[0]!=0:
                        
                        path_roi=graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path,band,'ROI',label,id_cross=bandm,num_columns=2,save=True,plot=False)
                        
                        

print('valelinda')
