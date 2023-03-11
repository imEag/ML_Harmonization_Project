'''
Code used to delete subjects with more atypical power data by ROIs and independent components.
'''

from xml.dom.expatbuilder import ParseEscape
import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd
import warnings
import collections
from Funciones import dataframe_long_roi, dataframe_long_components,estadisticos_demograficos,removing_outliers
from Funciones import columns_powers_ic,columns_powers_rois
'Eliminating outliers in powers by independent components'

#Load data
path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' 
data_Comp=pd.read_feather(r'{path}\Datosparaorganizardataframes\BasesdeDatosFiltradas_componenteporcolumnas.feather'.format(path=path))
data_Comp=data_Comp.drop(columns='index')


data_Comp_copy=removing_outliers(data_Comp,columns_powers_ic)# dataframe without outliers

#The dataframe of independent component powers is saved without atypical data
data_Comp_copy.to_feather('{path}\Datosparaorganizardataframes\BasesdeDatosFiltradas_componenteporcolumnas_sin_atipicos.feather'.format(path=path))
print('\nFinalización de eliminación de datos atipicos de componentes')

'Eliminating outliers in powers by ROIs'
#Load data
data_roi=pd.read_feather(r'{path}\Datosparaorganizardataframes\BasesdeDatosFiltradas_ROIporcolumnas.feather'.format(path=path))
data_roi=data_roi.drop(columns='index')

data_roi_copy=removing_outliers(data_roi,columns_powers_rois)# dataframe without outliers

#The dataframe of ROIs powers is saved without atypical data
data_roi_copy.to_feather('{path}\Datosparaorganizardataframes\BasesdeDatosFiltradas_ROIporcolumnas_sin_atipicos.feather'.format(path=path))
print('\nFinalización de eliminación de datos atipicos de ROIs')


data_roi_copy['group'].replace({'CTR':'Control','G4':'Control','G3':'DTA'}, inplace=True)
data_Comp_copy['group'].replace({'CTR':'Control','G4':'Control','G3':'DTA'}, inplace=True)

#Dataframes are saved by ROI and components for graphics.
dataframe_long_roi(data_roi_copy,'Power',columns=columns_powers_rois,name="data_long_power_roi_without_oitliers",path=path)
dataframe_long_components(data_Comp_copy,'Power',columns=columns_powers_ic,name="data_long_power_components_without_oitliers",path=path)

#Distribucion de edad, escolaridad y educación
estadisticos_demograficos(data_Comp_copy,'componentes',path)
estadisticos_demograficos(data_roi_copy,'ROIs',path)

