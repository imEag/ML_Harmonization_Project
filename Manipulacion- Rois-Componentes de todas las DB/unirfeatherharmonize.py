from os import remove
import pandas as pd 
import warnings
#from Graficos_power_sl_coherencia_entropia_cross import graphics
warnings.filterwarnings("ignore")
import os 
import tkinter as tk
from tkinter.filedialog import askdirectory

tk.Tk().withdraw() # part of the import if you are not using other tkinter functions
path = askdirectory() 
#path = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD\Datosparaorganizardataframes/' 
A = 'G1'
B = ''
m = ['power','sl','cohfreq','entropy']
#s = ['roi','ic']
s=['ic']
h = ['neuroHarmonize','sovaharmony']
group = 0
for space in s:
    for neuro in h:
        if group == None:
            data_power=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{B}\Data_complete_{space}_{neuro}_{B}_power.feather'.replace('\\','/'))
            data_sl=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{B}\Data_complete_{space}_{neuro}_{B}_sl.feather'.replace('\\','/'))
            data_cohfreq=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{B}\Data_complete_{space}_{neuro}_{B}_cohfreq.feather'.replace('\\','/'))
            data_entropy=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{B}\Data_complete_{space}_{neuro}_{B}_entropy.feather'.replace('\\','/'))
            data_crossfreq=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{B}\Data_complete_{space}_{neuro}_{B}_crossfreq.feather'.replace('\\','/'))
            data=pd.concat([data_power,data_sl,data_cohfreq,data_entropy,data_crossfreq], axis=1,)
            data = data.T.drop_duplicates().T
            new_name = 'Data_complete_'+space+'_'+neuro
            path_integration = fr'{path}\{neuro}\integration\{space}'
            os.makedirs(path_integration,exist_ok=True)
            data.reset_index(drop=True).to_feather('{path}\{name}.feather'.format(path=path_integration,name=new_name))
            #remove(rf'{path}\Data_complete_{space}_{neuro}_power.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_sl.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_cohfreq.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_entropy.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_crossfreq.feather')
        else:
            data_power=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{A+B}\Data_complete_{space}_{neuro}_{A+B}_power.feather'.replace('\\','/'))
            data_sl=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{A+B}\Data_complete_{space}_{neuro}_{A+B}_sl.feather'.replace('\\','/'))
            data_cohfreq=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{A+B}\Data_complete_{space}_{neuro}_{A+B}_cohfreq.feather'.replace('\\','/'))
            data_entropy=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{A+B}\Data_complete_{space}_{neuro}_{A+B}_entropy.feather'.replace('\\','/'))
            data_crossfreq=pd.read_feather(rf'{path}\{neuro}\complete\{space}\{A+B}\Data_complete_{space}_{neuro}_{A+B}_crossfreq.feather'.replace('\\','/'))
            data=pd.concat([data_power,data_sl,data_cohfreq,data_entropy,data_crossfreq], axis=1,)
            data = data.T.drop_duplicates().T
            new_name = 'Data_complete_'+space+'_'+neuro+'_'+A+B
            path_integration = fr'{path}\{neuro}\integration\{space}\{A+B}'
            os.makedirs(path_integration,exist_ok=True)
            data.reset_index(drop=True).to_feather('{path}\{name}.feather'.format(path=path_integration,name=new_name))
            #remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_power.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_sl.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_cohfreq.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_entropy.feather')
            #remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_crossfreq.feather')
