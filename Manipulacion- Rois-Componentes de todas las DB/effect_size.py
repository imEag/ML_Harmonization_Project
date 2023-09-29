from graficos_data_harmonized import stats_pair,stats_pair_database,effect_size_inside_DB
import pandas as pd
import os

A='G1'
B='G2'
path=r'E:\BIOMARCADORES\derivatives\data_long\IC'
task=['CE','OE']
metric=['power','irasa']
configuration=['54x10']
conf=False
for m in metric:
    for t in task:
        for con in configuration:
            file = rf'tabla_effectsize_{con}_{t}_{m}_{A}_{B}.xlsx'.format(A,B,m,t,con)
            writer = pd.ExcelWriter(os.path.join(path,file), engine='xlsxwriter')

            data_power=pd.read_feather(fr'{path}\data_{t}_{m}_long_BIOMARCADORES_{con}_components.feather'.format(m,t,con).replace('\\','/'))
            #data_power=pd.read_feather(fr'{path}\data_BIOMARCADORES_{t}_long_{m}_{con}_components.feather'.format(m,t,con).replace('\\','/'))
            data={m:data_power}
            for metric in data.keys():
                data=data[metric]
                table=stats_pair(data,metric,'Component',A,B)
            table.to_excel(writer,sheet_name=metric)
            writer._save()
        