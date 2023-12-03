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
s = ['IC']#['roi','ic']
h = ['sova']#['harmonized','sova']
group = 0
# for space in s:
#     for neuro in h:
#         if B == 'G2':
#             C = A+B
#         else:
#             C = A
#         data_power=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_power_{space}_{neuro}_{A+B}.feather')
#         data_sl=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_sl_{space}_{neuro}_{A+B}.feather')
#         data_cohfreq=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_coherence_{space}_{neuro}_{A+B}.feather')
#         data_entropy=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_entropy_{space}_{neuro}_{A+B}.feather')
#         data_crossfreq=pd.read_feather(rf'{path}\{neuro}\long\{space}\{C}\data_long_crossfreq_{space}_{neuro}_{A+B}.feather')
#         data=pd.concat([data_power,data_sl,data_cohfreq,data_entropy,data_crossfreq], axis=1,)
#         data = data.dropna()
#         data = data.T.drop_duplicates().T
#         data = data.loc[:,~data.columns.duplicated()]
#         new_name = 'Data_long_complete_'+space+'_'+neuro+'_'+A+B
#         data.reset_index(drop=True).to_feather(fr'{path}\{neuro}\{new_name}.feather')

configs=['paper','cresta','openBCI']
spaces=['IC']
tasks=['CE']
names=['BIOMARCADORES']
for config in configs:
    for space in spaces:
        for task in tasks:
            for name in names:
                #data_power=pd.read_feather(rf'{path}\data_long\{space}\data_{task}_Power_long_{name}_{config}_components.feather'.replace('\\','/'))
                #data_sl=pd.read_feather(rf'{path}\data_long\{space}\data_{task}_SL_long_{name}_{config}_components.feather'.replace('\\','/'))
                #data_crossfreq=pd.read_feather(rf'{path}\data_long\{space}\data_{name}_{task}_long_Cross Frequency_{config}_components.feather'.replace('\\','/'))
                data_power=pd.read_feather(rf'{path}\data_columns\{space}\data_{name}_{task}_columns_power_{config}_components.feather'.replace('\\','/'))
                data_sl=pd.read_feather(rf'{path}\data_columns\{space}\data_{name}_{task}_columns_sl_{config}_components.feather'.replace('\\','/'))
                data_crossfreq=pd.read_feather(rf'{path}\data_columns\{space}\data_{name}_{task}_columns_crossfreq_{config}_components.feather'.replace('\\','/'))
                
                data=pd.concat([data_power,data_sl], axis=0, ignore_index=True)
                data = pd.merge(left=data_power, right=data_sl, how='left')
                # data = data.dropna()
                # data = data.drop_duplicates()
                # data = data.loc[:,~data.columns.duplicated()]
                new_name = f'Data_long_complete_{name}_{config}_{space}_{task}'
                #data.reset_index(drop=True).to_feather(fr'{path}\{new_name}.feather')
                data.reset_index(drop=True).to_csv(fr'{path}\{new_name}.csv')