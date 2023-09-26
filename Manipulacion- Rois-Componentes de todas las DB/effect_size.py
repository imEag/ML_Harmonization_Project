from graficos_data_harmonized import stats_pair,stats_pair_database,effect_size_inside_DB
import pandas as pd

A='G2'
B='G1'
path='/media/gruneco-server/ADATA HD650/BIOMARCADORES/derivatives/data_long'
file = rf'tabla_effectsize_{A}_{B}.xlsx'.format(A,B)
writer = pd.ExcelWriter(path+file)
data_power=pd.read_feather(fr'{path}\data_OE_power_long_BIOMARCADORES_54x10_components.feather'.replace('\\','/'))
data={'power':data_power}
for metric in data.keys():
    data=data[metric]
    table=stats_pair(data,metric,'Component',A,B)
table.to_excel(writer,sheet_name='irasa')
writer._save()
writer.close()