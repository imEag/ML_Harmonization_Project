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
B = 'Control'
m = ['power','sl','cohfreq','entropy']
s = ['roi','ic']
h = ['harmonized','sova']
group = 0
for space in s:
    for neuro in h:
        data_g1=pd.read_feather(rf'{path}\Data_long_complete_{space}_{neuro}_G1Control.feather')
        data_dta=pd.read_feather(rf'{path}\data_long_sl_{space}_{neuro}_DTAControl.feather')
        data_g1g2=pd.read_feather(rf'{path}\data_long_sl_{space}_{neuro}_G1G2.feather')
        data=pd.concat([data_g1,data_dta,data_g1g2], axis=1,)
        data = data.dropna()
        data = data.T.drop_duplicates().T
        data = data.loc[:,~data.columns.duplicated()]
        new_name = 'Data_long_complete_'+space
        data.reset_index(drop=True).to_feather('{path}\{name}.feather'.format(path=path,name=new_name))
