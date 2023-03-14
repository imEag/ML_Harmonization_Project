'''Code used to filter the BIOMARCADORES database by components of the G2 and CTR groups.
They have the same number of visits but do not correspond to the same visits.
The ICC is calculated with these data '''

import numpy as np
import pandas as pd 
import collections
import scipy.io
from tokenize import group
import pingouin as pg
from scipy import stats
import dataframe_image as dfi
import warnings
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")

datos1=pd.read_feather(r"Reliability\Data_csv_Powers_Componentes-Channels\longitudinal_data_powers_long_CE_components.feather") 
datos2=pd.read_feather(r"Reliability\Data_csv_Powers_Componentes-Channels\longitudinal_data_powers_long_CE_norm_components.feather")
datos=pd.concat((datos1, datos2))#Original Data
path=r'C:\Users\valec\OneDrive - Universidad de Antioquia\Resultados_Armonizacion_BD' 
N_BIO=pd.read_excel('{path}\Datosparaorganizardataframes\Demograficosbiomarcadores - copia.xlsx'.format(path=path))
N_BIO = N_BIO.rename(columns={'Codigo':'Subject','Edad en la visita':'age','Sexo':'sex','Escolaridad':'education','Visita':'Session'})
N_BIO=N_BIO.drop(['MMSE', 'F', 'A', 'S'], axis=1)
N_BIO['Subject']=N_BIO['Subject'].replace({'_':''}, regex=True)#Quito el _ y lo reemplazo con '' 


def pair_data(datos,components):
    #datos=datos.drop(datos[datos['Session']=='V4P'].index)#Borrar datos
    datos['Session']=datos['Session'].replace({'VO':'V0','V4P':'V4'})
    groups=['CTR','G2'] # Pair groups 
    datos=datos[datos.Components.isin(components) ] # Only data of components select
    datos=datos[datos.Group.isin(groups) ] #Only data of CTR y G2
    visitas=['V0','V1','V2','V3']
    #Script for drop subjects without four sessions select
    for i in groups:
        g=datos[datos['Group']==i]
        sujetos=g['Subject'].unique()
        print('Cantidad de sujetos de '+i+': ', len(sujetos))
        k=0
        for j in sujetos:
            s=g[g['Subject']==j]
            if len(s['Session'].unique()) !=4:
                k=k+1
                datos=datos.drop(datos[datos['Subject']==j].index)
            if len(s['Session'].unique()) ==4:
                v=s['Session'].unique()
                for vis in range(len(visitas)):
                    datos.loc[(datos.Subject==j)&(datos.Session==v[vis]),'Session']=visitas[vis]     
        print('Sujetos a borrar:', k)
        print('Sujetos a analizar con 4 visitas: ',len(sujetos)-k)
    for i in groups:
        g=datos[datos['Group']==i]
        print('Cantidad de sujetos al filtrar '+ i+': ',len(g['Subject'].unique()))
    #datos['Group']=datos['Group'].replace({'CTR':'Control','G2':'Control'})
    print('Visitas de los sujetos: ',datos['Session'].unique())
    return datos

components=['C14', 'C15','C18', 'C20', 'C22','C23', 'C24', 'C25' ] #Neuronal components
datos=pair_data(datos,components) #Datos filtrados
N_BIO=N_BIO[N_BIO.Session.isin(['V0'])]
N_BIO=N_BIO.drop(['Session'], axis=1)
datos=pd.merge(datos,N_BIO)
#Datos estadisticos de la edad, escolaridad y sexo
d_p=datos[datos.Session.isin(['V0'])&datos.Bands.isin(['delta'])&datos.Components.isin(['C14'])&datos.Stage.isin(['Normalized data'])]
datos_estadisticos=d_p.groupby(['Group']).describe(include='all')
table=datos_estadisticos.loc[:,[('age','count'),('age','mean'),('age','std'),('education','count'),('education','mean'),('education','std'),('sex','count'),('sex','top'),('sex','freq')]]
datos_estadisticos_juntos=d_p.groupby(['Stage']).describe(include='all')
table2=datos_estadisticos_juntos.loc[:,[('age','count'),('age','mean'),('age','std'),('education','count'),('education','mean'),('education','std'),('sex','count'),('sex','top'),('sex','freq')]]
#dfi.export(table, 'Reliability\Tabla_edad_escolaridad_sexo_separadoreliability.png')
#Pruebas estadisticas 
print('\nNormalidad en edad')
print(pg.normality(data=d_p, dv='age', group='Group',method='shapiro'))
print('\nNormalidad en educacion')
print(pg.normality(data=d_p, dv='education', group='Group',method='shapiro'))
print('\nDiferencias en edad entre grupos U-test')
print(mannwhitneyu(d_p[d_p.Group.isin(['CTR'])]['age'], d_p[d_p.Group.isin(['G2'])]['age']))
print('\nDiferencias en escolaridad entre grupos U-test')
print(mannwhitneyu(d_p[d_p.Group.isin(['CTR'])]['education'], d_p[d_p.Group.isin(['G2'])]['education']))
## Diferencias estadisticas entre CTR y G2 en cuanto la edad y escolaridad

#Tabla edad escolaridad y genero y valores p asociados a las pruebas 

# ANOVA mix
# print('Anova mix')
# aov = pg.mixed_anova(data = datos, dv = 'Powers', between = 'Group', within = 'Session',subject = 'Subject')
# pg.print_table(aov)

#U test
print('Test U')
ap=pg.mwu(datos[datos['Group']=='CTR']['Powers'],datos[datos['Group']=='G2']['Powers'])
pg.print_table(ap)

bandas=datos['Bands'].unique()
Stage=datos['Stage'].unique()
# icc_value = pd.DataFrame(columns=['ICC','F','df1','df2','pval','CI95%'])
# G=['CTR','G2']
# for st in Stage:
#     d_stage=datos[datos['Stage']==st] 
#     for g in G:
#         d_group=d_stage[d_stage['Group']==g]
#         dic={}
#         icc_comp=[]
#         for comp in components:
#             d_comp=d_group[d_group['Components']==comp]
#             visits=list(d_comp['Session'].unique())
#             matrix_c=pd.DataFrame(columns=['index','Session', 'Power','Bands','Group','Stage','Subject']) #Se le asigna a un dataframe los datos d elas columnas
#             subjects=d_comp['Subject'].unique() 
#             for vis in visits:
#                 matrix_s=pd.DataFrame(columns=['index','Session', 'Power','Bands','Group','Stage','Subject'])
#                 power=d_comp[d_comp['Session']==vis]['Powers'].tolist()
#                 n_vis=[vis]*len(power)
#                 matrix_s['Session']=n_vis
#                 matrix_s['Power']=power  
#                 matrix_s['Group']=d_comp[d_comp['Session']==vis]['Group'].tolist()
#                 matrix_s['Bands']=d_comp[d_comp['Session']==vis]['Bands'].tolist()
#                 matrix_s['Stage']=d_comp[d_comp['Session']==vis]['Stage'].tolist()
#                 matrix_s['Subject']=d_comp[d_comp['Session']==vis]['Subject'].tolist()

#                 matrix_c=matrix_c.append(matrix_s, ignore_index = True)            
            
#             index=list(np.arange(0,len(n_vis),1))*len(visits)
#             matrix_c['index']=index

#             for i,ban in enumerate(bandas):
#                 fil_bands=matrix_c['Bands']==ban
#                 filter=matrix_c[fil_bands]
#                 icc=pg.intraclass_corr(data=filter, targets='index', raters='Session', ratings='Power').round(6)
#                 #icc3=icc
#                 icc3 = icc[icc['Type']=='ICC3k']
#                 icc3 = icc3.set_index('Type')
#                 # print(filter['Stage'])
#                 icc3['Stage']=st #filter['Stage'][i]
#                 icc3['Group']=g #filter['Group'][i]
#                 icc3['Bands']=ban
#                 icc3['Components']=comp
#                 icc_value=icc_value.append(icc3,ignore_index=True)

#         icc_value.append(icc_value)
#     icc_value.append(icc_value)
#print(icc_value)
#icc_value.to_csv(r'Reliability\ICC_values_csv\icc_values_Components_G2-CTR.csv',sep=';')
#matrix_c.to_csv(r'sovaharmony\Reproducibilidad\icc_values_G2-CTR_test.csv',sep=';') 


## ICC sin separar por grupos 
icc_value = pd.DataFrame(columns=['ICC','F','df1','df2','pval','CI95%'])
d_stage=datos[datos['Stage']=='Preprocessed data'] 
d_group=d_stage.copy()
dic={}
icc_comp=[]
st='Preprocessed data'
for comp in components:
    d_comp=d_group[d_group['Components']==comp]
    visits=list(d_comp['Session'].unique())
    matrix_c=pd.DataFrame(columns=['index','Session', 'Power','Bands','Group','Stage','Subject']) #Se le asigna a un dataframe los datos d elas columnas
    subjects=d_comp['Subject'].unique() 
    for vis in visits:
        matrix_s=pd.DataFrame(columns=['index','Session', 'Power','Bands','Group','Stage','Subject'])
        power=d_comp[d_comp['Session']==vis]['Powers'].tolist()
        n_vis=[vis]*len(power)
        matrix_s['Session']=n_vis
        matrix_s['Power']=power  
        matrix_s['Bands']=d_comp[d_comp['Session']==vis]['Bands'].tolist()
        matrix_s['Stage']=d_comp[d_comp['Session']==vis]['Stage'].tolist()
        matrix_s['Subject']=d_comp[d_comp['Session']==vis]['Subject'].tolist()
        matrix_c=matrix_c.append(matrix_s, ignore_index = True)            
    index=list(np.arange(0,len(n_vis),1))*len(visits)
    matrix_c['index']=index
    for i,ban in enumerate(bandas):
        fil_bands=matrix_c['Bands']==ban
        filter=matrix_c[fil_bands]
        icc=pg.intraclass_corr(data=filter, targets='index', raters='Session', ratings='Power').round(6)
        #icc3=icc
        icc3 = icc[icc['Type']=='ICC3k']
        icc3 = icc3.set_index('Type')
        # print(filter['Stage'])
        icc3['Stage']=st #filter['Stage'][i]
        icc3['Group']='Healthy'#filter['Group'][i]
        icc3['Bands']=ban
        icc3['Neural Component']=comp
        icc_value=icc_value.append(icc3,ignore_index=True)
icc_value.append(icc_value)
table=pd.pivot_table(icc_value, values='ICC', index=['Neural Component'],columns=['Bands'])
#print(icc_value)
writer = pd.ExcelWriter('Reliability\ICC_values_csv\ICC_components_sin_separar.xlsx')
icc_value.to_excel(writer ,sheet_name='Component_graphic')
table.to_excel(writer ,sheet_name='Component')
writer.save()
writer.close()  
