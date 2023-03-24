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
from Funciones import columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_powers_ic
#from Graficos_power_sl_coherencia_entropia_cross import graphics
warnings.filterwarnings("ignore")

#path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' #Cambia dependieron de quien lo corra
path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD'

#Data loading ic
#data_roi_sova=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_G2G1.feather'.format(path=path))
#data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_G2G1.feather'.format(path=path))
##data_roi_sova=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony.feather'.format(path=path))
##data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize.feather'.format(path=path))
#space = 'ic'

#Data loading roi
data_roi_sova=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_G2G1.feather'.format(path=path))
##data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize.feather'.format(path=path))
data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_G2G1.feather'.format(path=path))
space = 'roi'

#Other
#data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_All(SRM_CHBMP_G1_G2).feather'.format(path=path))
#data_roi_harmo=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_All(G1_G2).feather'.format(path=path))

columns_All=data_roi_harmo.copy().columns.tolist()
for i in ['participant_id', 'group', 'visit', 'condition','database','sex','MM_total','FAS_F','FAS_S','FAS_A','education','age']:
   columns_All.remove(i)
if space == 'ic':
    columns = [columns_All[:64],columns_All[64:128],columns_All[128:192],columns_All[192:256],columns_All[256:]]
elif space == 'roi':
    columns = [columns_All[:32],columns_All[32:64],columns_All[64:96],columns_All[96:128],columns_All[128:]]

#Columnas de cross frequency
#if space == 'roi':
#    DUQUE_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_crossfreq_columns_ROI_DUQUE.feather'.format(path=path))
#    columns_cross_roi=DUQUE_cr.columns.tolist()
#    columns = [columns_powers_rois,columns_SL_roi,columns_coherence_roi,columns_entropy_rois]
#elif space == 'ic':
#    DUQUE_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_crossfreq_columns_components_DUQUE.feather'.format(path=path))
#    columns_cross_roi=DUQUE_cr.columns.tolist()
#    columns = [columns_powers_ic,columns_SL_ic,columns_coherence_ic,columns_entropy_ic]
#for i in ['participant_id', 'group', 'visit', 'condition','database']:
    #columns_cross_roi.remove(i)

A = 'G2'
B = 'G1'


for i,value in enumerate([data_roi_sova,data_roi_harmo]):
    d_B_roi=value
    if i==0:
        label="_sova"
    else:
        label="_harmonized"
    #New dataframes from ROIs
    #Dataframes are saved by ROI and components for graphics.
    if space == 'roi':
        #Power
        dataframe_long_roi(d_B_roi,'Power',columns=columns[0],name="data_long_power_{space}_without_oitliers{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #SL
        dataframe_long_roi(d_B_roi,type='SL',columns=columns[1],name="data_long_sl_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #Coherencia
        dataframe_long_roi(d_B_roi,type='Coherence',columns=columns[2],name="data_long_coherence_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #Entropia
        dataframe_long_roi(d_B_roi,type='Entropy',columns=columns[3],name="data_long_entropy_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #Cross frequency
        dataframe_long_cross_roi(d_B_roi,type='Cross Frequency',columns=columns[4],name="data_long_crossfreq_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
    else:
        #Power
        dataframe_long_components(d_B_roi,'Power',columns=columns[0],name="data_long_power_{space}_without_oitliers{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #SL
        dataframe_long_components(d_B_roi,type='SL',columns=columns[1],name="data_long_sl_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #Coherencia
        dataframe_long_components(d_B_roi,type='Coherence',columns=columns[2],name="data_long_coherence_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #Entropia
        dataframe_long_components(d_B_roi,type='Entropy',columns=columns[3],name="data_long_entropy_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)
        #Cross frequency
        dataframe_long_cross_ic(d_B_roi,type='Cross Frequency',columns=columns[4],name="data_long_crossfreq_{space}{label}_{g}".format(label=label,g=str(A+B),space=space),path=path)

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
            path_complete='{path}\Graficos_armonizacion_sova_harmo\Graficos_{type}\{id}\{name_band}_{type}_{id}_{id2}_{space}_{group}.png'.format(path=path,name_band=name_band,id=id,id2=id2,type=type,group=str(A+B),space=space)  
        else:
            path_complete='{path}\Graficos_armonizacion_sova_harmo\Graficos_{type}\{id}\{name_band}_{id_cross}_{type}_{id}_{id2}_{space}_{group}.png'.format(path=path,name_band=name_band,id=id,id2=id2,type=type,id_cross=id_cross,group=str(A+B),space=space)
        plt.savefig(path_complete)
    plt.close()
    return path_complete

def text_format(val,value):
    if value==0.2: #Cambie el 0.05 por 0.2 y el lightgreen por lightblue
        color = 'lightblue' if val <0.2 else 'white'
    if value==0.7:
        color = 'lightgreen' if np.abs(val)>=0.7 else 'white'
    if value==0.0:
        color = 'lightblue' if np.abs(val)<=0.05 else 'white'
#    elif value==0.8:
#        if val >=0.7 and val<0.8:
#            color = 'salmon'
#        elif val >=0.8:
#            color = 'lightblue' 
#        else:
#            color='white'

    return 'background-color: %s' % color

def stats_pair(data,metric,space):
    
    data_DB=data.copy()
    if metric!='Cross Frequency':
        ez=data_DB.groupby([space,'Band']).apply(lambda data_DB:pg.compute_effsize(data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric])).to_frame()
        ez=ez.rename(columns={0:'effect size'})
        ez['A']=A
        ez['B']=B
        ez['Prueba']='effect size'
        #cv
        std=data_DB.groupby([space,'Band']).apply(lambda data_DB:np.std(np.concatenate((data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric]),axis=0))).to_frame()
        std=std.rename(columns={0:'cv'})
        std['A']=A
        std['B']=B
        std['Prueba']='cv'

        table_concat=pd.concat([ez,std],axis=0)
        table=pd.pivot_table(table_concat,values=['effect size','cv'],columns=['Prueba'],index=[space,'Band','A', 'B'])
    else:
        ez=data_DB.groupby([space,'Band','M_Band']).apply(lambda data_DB:pg.compute_effsize(data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric])).to_frame()
        ez=ez.rename(columns={0:'effect size'})
        ez['A']=A
        ez['B']=B
        ez['Prueba']='effect size'
        #cv
        std=data_DB.groupby([space,'Band','M_Band']).apply(lambda data_DB:np.std(np.concatenate((data_DB[data_DB['group']==A][metric],data_DB[data_DB['group']==B][metric]),axis=0))).to_frame()
        std=std.rename(columns={0:'cv'})
        std['A']=A
        std['B']=B
        std['Prueba']='cv'

        table_concat=pd.concat([ez,std],axis=0)
        table=pd.pivot_table(table_concat,values=['effect size','cv'],columns=['Prueba'],index=[space,'Band','M_Band','A', 'B'])
    table=table.reset_index()
    table=table.style.applymap(text_format,value=0.7,subset=['effect size']).applymap(text_format,value=0.0,subset=['cv'])
    
    #dfi.export(table, path_complete)
    return table

#labels=['_harmonized']
labels=['_sova','_harmonized']
for label in labels:
    data_p_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_power_{space}_without_oitliers{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
    data_sl_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_sl_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
    data_c_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_coherence_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
    data_e_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_entropy_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space))
    data_cr_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_long_crossfreq_{space}{label}_{g}.feather'.format(label=label,path=path,g=str(A+B),space=space)) 
    datos_roi={'Power':data_p_roi,'SL':data_sl_roi,'Coherence':data_c_roi,'Entropy':data_e_roi,'Cross Frequency':data_cr_roi}
    bands= data_sl_roi['Band'].unique()
    bandsm= data_cr_roi['M_Band'].unique()  

    #filename = r"{path}\Graficos_armonizacion_sova_harmo\tabla_effectsize{label}.xlsx".format(path=path,label=label)
    filename = r"{path}\Graficos_armonizacion_sova_harmo\tabla_effectsize_{space}_{group}{label}.xlsx".format(path=path,label=label,group=str(A+B),space=space)
    writer = pd.ExcelWriter(filename)
  
    for metric in datos_roi.keys():
       d_roi=datos_roi[metric]
       if space == 'roi':
           table=stats_pair(d_roi,metric,'ROI')
       else:
           table=stats_pair(d_roi,metric,'Component')
       table.to_excel(writer ,sheet_name=metric)
    writer.save()
    writer.close() 

    for metric in datos_roi.keys():
        for band in bands:
            d_roi=datos_roi[metric]
            d_banda_roi=d_roi[d_roi['Band']==band]
            if metric!='Cross Frequency':  
                print(str(band)+' '+str(metric)) 
                if space == 'roi':
                    path_roi=graphics(d_banda_roi,metric,path,band,'ROI',label,num_columns=2,save=True,plot=False)
                else:
                    path_roi=graphics(d_banda_roi,metric,path,band,'IC',label,num_columns=2,save=True,plot=False)
                
            else:
                #pass
                for bandm in list(d_banda_roi['M_Band'].unique()):  
                    print(str(band)+' '+str(metric)+' '+str(bandm)) 
                    if d_banda_roi[d_banda_roi['M_Band']==bandm]['Cross Frequency'].iloc[0]!=None:
                        if space == 'roi':
                            path_roi=graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path,band,'ROI',label,id_cross=bandm,num_columns=2,save=True,plot=False)
                        else:
                            path_roi=graphics(d_banda_roi[d_banda_roi['M_Band']==bandm],'Cross Frequency',path,band,'IC',label,id_cross=bandm,num_columns=2,save=True,plot=False)
