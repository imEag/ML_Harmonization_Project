"""
Code to manipulate the dataframes of the SL, coherence, entropy and cross frequency files
of each database and then use them for graphing.
 """

from cmath import nan
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display_html, display
import collections
from Funciones import dataframe_long_roi,dataframe_long_components,dataframe_componentes_deseadas,dataframe_long_cross_ic,dataframe_long_cross_roi
from Funciones import columns_SL_roi,columns_coherence_roi,columns_entropy_rois,columns_powers_rois
from Funciones import columns_SL_ic,columns_coherence_ic,columns_entropy_ic,columns_powers_ic
from Funciones import ver_datos_vacios

path=r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD' #Cambia dependieron de quien lo corra

"Load data"

#Data by ROIs with demographic data

#Power
data_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\BasesdeDatosFiltradas_ROIporcolumnas_sin_atipicos.feather'.format(path=path))

# SL Dataframes
SRM_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_sl_columns_ROI_SRM.feather'.format(path=path))
CHBMP_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_sl_columns_ROI_CHBMP.feather'.format(path=path))
BIO_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_sl_columns_ROI_BIOMARCADORES.feather'.format(path=path))
DUQUE_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_sl_columns_ROI_DUQUE.feather'.format(path=path))

#Coherence Dataframes
SRM_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_cohfreq_columns_ROI_SRM.feather'.format(path=path))
CHBMP_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_cohfreq_columns_ROI_CHBMP.feather'.format(path=path))
BIO_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_cohfreq_columns_ROI_BIOMARCADORES.feather'.format(path=path))
DUQUE_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_cohfreq_columns_ROI_DUQUE.feather'.format(path=path))

#Entropy Dataframes
SRM_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_entropy_columns_ROI_SRM.feather'.format(path=path))
CHBMP_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_entropy_columns_ROI_CHBMP.feather'.format(path=path))
BIO_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_entropy_columns_ROI_BIOMARCADORES.feather'.format(path=path))
DUQUE_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_entropy_columns_ROI_DUQUE.feather'.format(path=path))

#Cross frequency Dataframes
SRM_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_crossfreq_columns_ROI_SRM.feather'.format(path=path))
CHBMP_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_crossfreq_columns_ROI_CHBMP.feather'.format(path=path))
BIO_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_crossfreq_columns_ROI_BIOMARCADORES.feather'.format(path=path))
DUQUE_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_crossfreq_columns_ROI_DUQUE.feather'.format(path=path))
columns_cross_roi=DUQUE_cr.columns.tolist()

for i in ['participant_id', 'group', 'visit', 'condition','database']:
    columns_cross_roi.remove(i)

'''The power data by ROIs is merged with the demographic data, 
first merged for each database and then all the databases are concatenated'''

#SRM
d_SRM=pd.merge(left=data_roi[data_roi['database']=='SRM'],right=SRM_sl)
d_SRM=pd.merge(d_SRM,SRM_c)
d_SRM=pd.merge(d_SRM,SRM_e)
d_SRM=pd.merge(d_SRM,SRM_cr)
#CHBMP
mergeCHBMP=data_roi[data_roi['database']=='CHBMP']
mergeCHBMP.reset_index(inplace=True, drop=True)
mergeCHBMP=mergeCHBMP.drop(['visit'], axis=1)
d_CHBMP=pd.merge(mergeCHBMP,CHBMP_sl)
d_CHBMP=pd.merge(d_CHBMP,CHBMP_c)
d_CHBMP=pd.merge(d_CHBMP,CHBMP_e)
d_CHBMP=pd.merge(d_CHBMP,CHBMP_cr)
#BIOMARCADORES
mergeBIO=data_roi[data_roi['database']=='BIOMARCADORES']
mergeBIO.reset_index(inplace=True, drop=True)
d_BIO=pd.merge(mergeBIO,BIO_sl)
d_BIO=pd.merge(d_BIO,BIO_c)
d_BIO=pd.merge(d_BIO,BIO_e)
d_BIO=pd.merge(d_BIO,BIO_cr)
#DUQUE
mergeDUQUE=data_roi[data_roi['database']=='DUQUE']
mergeDUQUE.reset_index(inplace=True, drop=True)
mergeDUQUE=mergeDUQUE.drop(['visit'], axis=1)
DUQUE_sl=DUQUE_sl.drop(['group'], axis=1)
d_DUQUE=pd.merge(mergeDUQUE,DUQUE_sl)
DUQUE_c=DUQUE_c.drop(['group'], axis=1)
d_DUQUE=pd.merge(d_DUQUE,DUQUE_c)
DUQUE_e=DUQUE_e.drop(['group'], axis=1)
d_DUQUE=pd.merge(d_DUQUE,DUQUE_e)
DUQUE_cr=DUQUE_cr.drop(['group'], axis=1)
d_DUQUE=pd.merge(d_DUQUE,DUQUE_cr)

"Data concatenation: The dataframe contains power columns for ROIs, SL, coherence, entropy, cross frequency, etc"
d_B_roi=pd.concat([d_SRM,d_BIO,d_DUQUE,d_CHBMP])

d_B_roi['group'].replace({'CTR':'Control','G4':'Control','G3':'DTA'}, inplace=True)
d_B_roi.reset_index(inplace=True,drop=True)
ver_datos_vacios(d_B_roi)
d_B_roi.to_feather('{path}\Datosparaorganizardataframes\Data_complete_roi.feather'.format(path=path))
"Load data"
#Independent component data with demographic data

#Power
data_Comp=pd.read_feather(r'{path}\Datosparaorganizardataframes\BasesdeDatosFiltradas_componenteporcolumnas_sin_atipicos.feather'.format(path=path))

# SL Dataframes
SRM_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_sl_columns_components_SRM.feather'.format(path=path))
CHBMP_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_sl_columns_components_CHBMP.feather'.format(path=path))
BIO_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_sl_columns_components_BIOMARCADORES.feather'.format(path=path))
BIO_sl=dataframe_componentes_deseadas(BIO_sl,columnas=['participant_id','visit','group'])
DUQUE_sl=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_sl_columns_components_DUQUE.feather'.format(path=path))
DUQUE_sl=dataframe_componentes_deseadas(DUQUE_sl,columnas=['participant_id'])

#Coherence Dataframes
SRM_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_cohfreq_columns_components_SRM.feather'.format(path=path))
CHBMP_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_cohfreq_columns_components_CHBMP.feather'.format(path=path))
BIO_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_cohfreq_columns_components_BIOMARCADORES.feather'.format(path=path))
BIO_c=dataframe_componentes_deseadas(BIO_c,columnas=['participant_id','visit','group'])
DUQUE_c=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_cohfreq_columns_components_DUQUE.feather'.format(path=path))
DUQUE_c=dataframe_componentes_deseadas(DUQUE_c,columnas=['participant_id'])

#Entropy Dataframes
SRM_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_entropy_columns_components_SRM.feather'.format(path=path))
CHBMP_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_entropy_columns_components_CHBMP.feather'.format(path=path))
BIO_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_entropy_columns_components_BIOMARCADORES.feather'.format(path=path))
BIO_e=dataframe_componentes_deseadas(BIO_e,columnas=['participant_id','visit','group'])
DUQUE_e=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_entropy_columns_components_DUQUE.feather'.format(path=path))
DUQUE_e=dataframe_componentes_deseadas(DUQUE_e,columnas=['participant_id'])

#Cross frequency Dataframes
SRM_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resteyesc_crossfreq_columns_components_SRM.feather'.format(path=path))
CHBMP_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_protmap_crossfreq_columns_components_CHBMP.feather'.format(path=path))
BIO_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_CE_crossfreq_columns_components_BIOMARCADORES.feather'.format(path=path))
BIO_cr=dataframe_componentes_deseadas(BIO_cr,columnas=['participant_id','visit','group'])
DUQUE_cr=pd.read_feather(r'{path}\Datosparaorganizardataframes\data_resting_crossfreq_columns_components_DUQUE.feather'.format(path=path))
DUQUE_cr=dataframe_componentes_deseadas(DUQUE_cr,columnas=['participant_id'])

columns_cross_ic=DUQUE_cr.columns.tolist()
columns_cross_ic.remove('participant_id')
'''The power data by independet components is merged with the demographic data, 
first merged for each database and then all the databases are concatenated'''

#SRM
d_SRM=pd.merge(left=data_Comp[data_Comp['database']=='SRM'],right=SRM_sl)
d_SRM=pd.merge(d_SRM,SRM_c)
d_SRM=pd.merge(d_SRM,SRM_e)
d_SRM=pd.merge(d_SRM,SRM_cr)
#CHBMP
mergeCHBMP=data_Comp[data_Comp['database']=='CHBMP']
mergeCHBMP.reset_index(inplace=True, drop=True)
mergeCHBMP=mergeCHBMP.drop(['visit'], axis=1)
d_CHBMP=pd.merge(mergeCHBMP,CHBMP_sl)
d_CHBMP=pd.merge(d_CHBMP,CHBMP_c)
d_CHBMP=pd.merge(d_CHBMP,CHBMP_e)
d_CHBMP=pd.merge(d_CHBMP,CHBMP_cr)
#BIOMARCADORES
mergeBIO=data_Comp[data_Comp['database']=='BIOMARCADORES']
mergeBIO.reset_index(inplace=True, drop=True)
d_BIO=pd.merge(mergeBIO,BIO_sl)
d_BIO=pd.merge(d_BIO,BIO_c)
d_BIO=pd.merge(d_BIO,BIO_e)
d_BIO=pd.merge(d_BIO,BIO_cr)
#DUQUE
mergeDUQUE=data_Comp[data_Comp['database']=='DUQUE']
mergeDUQUE.reset_index(inplace=True, drop=True)
d_DUQUE=pd.merge(mergeDUQUE,DUQUE_sl)
d_DUQUE=pd.merge(d_DUQUE,DUQUE_c)
d_DUQUE=pd.merge(d_DUQUE,DUQUE_e)
d_DUQUE=pd.merge(d_DUQUE,DUQUE_cr)

"Data concatenation: The dataframe contains power columns for independent components, SL, coherence, entropy, cross frequency, etc"

d_B_com=pd.concat([d_SRM,d_BIO,d_DUQUE,d_CHBMP])#Union de todos los dataframes
d_B_com['group'].replace({'CTR':'Control','G4':'Control','G3':'DTA'}, inplace=True)
d_B_com.reset_index(inplace=True,drop=True)
ver_datos_vacios(d_B_com)
d_B_com.to_feather('{path}\Datosparaorganizardataframes\Data_complete_ic.feather'.format(path=path))

"""Conversion of dataframes to perform the different SL, coherence, entropy, cross frequency, etc. graphs"""

#New dataframes from ROIs
#Dataframes are saved by ROI and components for graphics.
dataframe_long_roi(d_B_roi,'Power',columns=columns_powers_rois,name="data_long_power_roi_without_oitliers",path=path)

#SL
dataframe_long_roi(d_B_roi,type='SL',columns=columns_SL_roi,name="data_long_sl_roi",path=path)
#Coherencia
dataframe_long_roi(d_B_roi,type='Coherence',columns=columns_coherence_roi,name="data_long_coherence_roi",path=path)
#Entropia
dataframe_long_roi(d_B_roi,type='Entropy',columns=columns_entropy_rois,name="data_long_entropy_roi",path=path)
#Cross frequency
dataframe_long_cross_roi(d_B_roi,type='Cross Frequency',columns=columns_cross_roi,name="data_long_crossfreq_roi",path=path)

#New dataframes from Independent components
dataframe_long_components(d_B_com,'Power',columns=columns_powers_ic,name="data_long_power_components_without_oitliers",path=path)

#SL
dataframe_long_components(d_B_com,type='SL',columns=columns_SL_ic,name="data_long_sl_components",path=path)
#Coherencia
dataframe_long_components(d_B_com,type='Coherence',columns=columns_coherence_ic,name="data_long_coherence_components",path=path)
#Entropia
dataframe_long_components(d_B_com,type='Entropy',columns=columns_entropy_ic,name="data_long_entropy_components",path=path)
#Cross frequency
dataframe_long_cross_ic(d_B_com,type='Cross Frequency',columns=columns_cross_ic,name="data_long_crossfreq_components",path=path)

