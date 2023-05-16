from os import remove
import pandas as pd 
import warnings
#from Graficos_power_sl_coherencia_entropia_cross import graphics
warnings.filterwarnings("ignore")
import tkinter as tk
from tkinter.filedialog import askdirectory

tk.Tk().withdraw() # part of the import if you are not using other tkinter functions
path = askdirectory() 
print("user chose", path, "for save graphs")
abc = ['DTA','Control','G1','G2']
A = 'G1'
B = 'G2'
s = ['roi','ic']
h = ['harmonized','sova']
group = 0
for space in s:
    for neuro in h:
        if B == 'G2':
            C = A+B
        else:
            C = A
        data_power=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_power_{space}_{neuro}_{A+B}.feather')
        data_sl=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_sl_{space}_{neuro}_{A+B}.feather')
        data_cohfreq=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_coherence_{space}_{neuro}_{A+B}.feather')
        data_entropy=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_entropy_{space}_{neuro}_{A+B}.feather')
        data_crossfreq=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_crossfreq_{space}_{neuro}_{A+B}.feather')
        data=pd.concat([data_power,data_sl,data_cohfreq,data_entropy,data_crossfreq], axis=1,)
        data = data.dropna()
        data = data.T.drop_duplicates().T
        data = data.loc[:,~data.columns.duplicated()]
        new_name = 'Data_long_complete_'+space+'_'+neuro+'_'+A+B
        data.reset_index(drop=True).to_feather(fr'{path}\{neuro}\{new_name}.feather')
