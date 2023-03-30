from os import remove
import pandas as pd 
import warnings
#from Graficos_power_sl_coherencia_entropia_cross import graphics
warnings.filterwarnings("ignore")

path = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD\Datosparaorganizardataframes/' 
A = ''
B = 'G1'
m = ['power','sl','cohfreq','entropy']
s = ['roi','ic']
h = ['neuroHarmonize','sovaharmony']
group = 0
for space in s:
    for neuro in h:
        if group == None:
            data_power=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_power.feather')
            data_sl=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_sl.feather')
            data_cohfreq=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_cohfreq.feather')
            data_entropy=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_entropy.feather')
            data_crossfreq=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_crossfreq.feather')
            data=pd.concat([data_power,data_sl,data_cohfreq,data_entropy,data_crossfreq], axis=1,)
            data = data.T.drop_duplicates().T
            new_name = 'Data_complete_'+space+'_'+neuro
            data.reset_index(drop=True).to_feather('{path}\{name}.feather'.format(path=path,name=new_name))
            remove(rf'{path}\Data_complete_{space}_{neuro}_power.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_sl.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_cohfreq.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_entropy.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_crossfreq.feather')
        else:
            data_power=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_power.feather')
            data_sl=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_sl.feather')
            data_cohfreq=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_cohfreq.feather')
            data_entropy=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_entropy.feather')
            data_crossfreq=pd.read_feather(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_crossfreq.feather')
            data=pd.concat([data_power,data_sl,data_cohfreq,data_entropy,data_crossfreq], axis=1,)
            data = data.T.drop_duplicates().T
            new_name = 'Data_complete_'+space+'_'+neuro+'_'+A+B
            data.reset_index(drop=True).to_feather('{path}\{name}.feather'.format(path=path,name=new_name))
            remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_power.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_sl.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_cohfreq.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_entropy.feather')
            remove(rf'{path}\Data_complete_{space}_{neuro}_{B+A}_crossfreq.feather')
