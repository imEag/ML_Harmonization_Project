import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from scipy import stats
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

'''Graphical analysis'''
def graphic_normality_box(data,x_value,y_value):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.boxplot(x=x_value, y=y_value, data=data, ax=ax)
    sns.swarmplot(x=x_value, y=y_value, data=data, color='black', alpha = 0.5, ax=ax);

'''Normality tests'''

#Shapiro-Wilk Test
def normality_Shapiro(data,name_data):
    '''data: data to test
    name_data: name of data to analyze
    '''
    stat,p=stats.shapiro(data)
    print('Shapiro normality test for '+name_data+'\n')
    print('stat=%.3f, p=%.3f' % (stat,p))
    if p>0.05:
        print("Probably Gaussian\n")
    else:
        print("Probably not Gaussian\n")


    '''
    return pg.normality(data=data, dv=dv, group=group,method='shapiro')
    data: pandas.DataFrame, series, list or 1D np.array
    Iterable. Can be either a single list, 1D numpy array, or a wide- or long-format pandas dataframe.

    dv: vstr (are de values)
    Dependent variable (only when data is a long-format dataframe).

    group:str
    Grouping variable (only when data is a long-format dataframe).
    '''
    

#D’Agostino’s K-squared test
def normality_Agostino(data,name_data):
    '''data: data to test
    name_data: name of data to analyze
    '''
    stat,p=stats.normaltest(data)
    print('D’Agostino’s K-squared normality test for '+name_data+'\n')
    print('stat=%.3f, p=%.3f' % (stat,p))
    if p>0.05:
        print("Probably Gaussian\n")
    else:
        print("Probably not Gaussian\n")

#Test de homocedasticidad

def homocedasticidad_N_groups_Shapiro(data,dv,group):
    '''
    Test equality of variance

    data:pandas.DataFrame, list or dict
    Iterable. Can be either a list / dictionnary of iterables or a wide- or long-format pandas dataframe.

    dv:str
    Dependent variable (only when data is a long-format dataframe).

    group:str
    Grouping variable (only when data is a long-format dataframe).

    Returns
    statspandas.DataFrame
    'W/T': Test statistic (‘W’ for Levene, ‘T’ for Bartlett)
    'pval': p-value
    'equal_var': True if data has equal variance
    '''
    pg.homoscedasticity(data=data, dv=dv, group=group, method='levene')

'''Anova'''
#Therefore, it is recommended to use a Welch ANOVA instead, followed by Games-Howell post-hoc tests, 
# which do not require the groups to have equal variances.
# If p-value<0.05 we can reject the null hypothesis that the values analyzed are equal between the groups

#All grooups
#pg.welch_anova(data=df, dv='name where are the values', between='groups to compare') 

#This function makes  two-by-two comparisons  which do not require the groups to have equal variances. It is non-parametric
#pg.pairwise_gameshowell(data=df, dv='name where are the values', between='groups to compare')

#Some theory https://statisticsbyjim.com/anova/welchs-anova-compared-to-classic-one-way-anova/

## Pruebas datos componentes

icc=['C14_rDelta', 'C14_rTheta', 'C14_rAlpha-1', 'C14_rAlpha-2',
       'C14_rBeta1', 'C14_rBeta2', 'C14_rBeta3', 'C14_rGamma', 'C15_rDelta',
       'C15_rTheta', 'C15_rAlpha-1', 'C15_rAlpha-2', 'C15_rBeta1',
       'C15_rBeta2', 'C15_rBeta3', 'C15_rGamma', 'C18_rDelta', 'C18_rTheta',
       'C18_rAlpha-1', 'C18_rAlpha-2', 'C18_rBeta1', 'C18_rBeta2',
       'C18_rBeta3', 'C18_rGamma', 'C20_rDelta', 'C20_rTheta', 'C20_rAlpha-1',
       'C20_rAlpha-2', 'C20_rBeta1', 'C20_rBeta2', 'C20_rBeta3', 'C20_rGamma',
       'C22_rDelta', 'C22_rTheta', 'C22_rAlpha-1', 'C22_rAlpha-2',
       'C22_rBeta1', 'C22_rBeta2', 'C22_rBeta3', 'C22_rGamma', 'C23_rDelta',
       'C23_rTheta', 'C23_rAlpha-1', 'C23_rAlpha-2', 'C23_rBeta1',
       'C23_rBeta2', 'C23_rBeta3', 'C23_rGamma', 'C24_rDelta', 'C24_rTheta',
       'C24_rAlpha-1', 'C24_rAlpha-2', 'C24_rBeta1', 'C24_rBeta2',
       'C24_rBeta3', 'C24_rGamma', 'C25_rDelta', 'C25_rTheta', 'C25_rAlpha-1',
       'C25_rAlpha-2', 'C25_rBeta1', 'C25_rBeta2', 'C25_rBeta3']

bandas=['Delta','Theta','Alpha-1','Alpha-2','Beta1','Beta2','Beta3','Gamma']
componentes=['C14', 'C15','C18', 'C20', 'C22','C23', 'C24', 'C25']
datos=pd.read_feather(r'Manipulacion- Rois-Componentes de todas las DB\Datosparaorganizardataframes\BasesdeDatosFiltradas_componenteporcolumnas.feather')
datos=datos.drop(columns='index')
datos['index'] = datos.index

groups=['Control', 'DTA']
resultados_comparacion = pd.DataFrame(columns=['Group','Component','Band','p-unc','Result','Databases evaluated'])
for j in groups:
    print('\nGrupo '+j)
    for i in icc:
        print('\nDatos de '+i)
        for k in bandas:
            if k in i:
                band=k
        for c in componentes:
            if c in i:
                componente=c
        if(datos[i].isnull().tolist()[0]==False):
            print('Normalidad')
            print(pg.normality(data=datos[datos['group'].isin([j])], dv=i, group='database',method='shapiro'))
            print('Varianza')
            print(pg.homoscedasticity(data=datos[datos['group'].isin([j])], dv=i, group='database',method='bartlett'))
            #EN general no estan dando con varianza igual
            print('Welch ANOVA')
            welch=pg.welch_anova(data=datos[datos['group'].isin([j])], dv=i, between='database')
            print(welch)
            valor=welch['p-unc'][0]
            if valor<0.05:
                print('Hay diferencias estadisticamente significativas entre las medias')
                print('No Comparables')
                result='group differences'
            else:
                print('No hay diferencias estadisticamente significativas entre las medias')
                print('Comparables')
                result='No group differences'

            print('Gameshowell test')
            a=pg.pairwise_gameshowell(data=datos[datos['group'].isin([j])], dv=i, between='database')
            print(a)
            databases=datos[datos['group'].isin(['Control'])]['database'].unique()
            b=a['A'].unique().tolist()+a['B'].unique().tolist()
            b=set(b)
            listToStr = ','.join([str(elem) for elem in b])
            nueva_fila = { 'Group':j,'Component':componente,'Band':band,'p-unc':valor,'Result':result,'Databases evaluated':listToStr} # creamos un diccionario
            resultados_comparacion = resultados_comparacion.append(nueva_fila, ignore_index=True)
resultados_comparacion.to_csv('Manipulacion- Rois-Componentes de todas las DB\Datosparaorganizardataframes\ResultadosWelchComponentes.csv',sep=';',index=False)

##Pruebas para los ROIs
datos=pd.read_feather(r'Manipulacion- Rois-Componentes de todas las DB\Datosparaorganizardataframes\BasesdeDatosFiltradas_ROIporcolumnas.feather')
datos=datos.drop(columns='index')
datos['index'] = datos.index

rois=['ROI_F_rDelta','ROI_C_rDelta', 'ROI_PO_rDelta', 'ROI_T_rDelta', 'ROI_F_rTheta',
       'ROI_C_rTheta', 'ROI_PO_rTheta', 'ROI_T_rTheta', 'ROI_F_rAlpha-1',
       'ROI_C_rAlpha-1', 'ROI_PO_rAlpha-1', 'ROI_T_rAlpha-1', 'ROI_F_rAlpha-2',
       'ROI_C_rAlpha-2', 'ROI_PO_rAlpha-2', 'ROI_T_rAlpha-2', 'ROI_F_rBeta1',
       'ROI_C_rBeta1', 'ROI_PO_rBeta1', 'ROI_T_rBeta1', 'ROI_F_rBeta2',
       'ROI_C_rBeta2', 'ROI_PO_rBeta2', 'ROI_T_rBeta2', 'ROI_F_rBeta3',
       'ROI_C_rBeta3', 'ROI_PO_rBeta3', 'ROI_T_rBeta3', 'ROI_F_rGamma',
       'ROI_C_rGamma', 'ROI_PO_rGamma', 'ROI_T_rGamma']
roi=['ROI_F', 'ROI_C','ROI_PO', 'ROI_T']

resultados_comparacion = pd.DataFrame(columns=['Group','ROI','Band','p-unc','Result','Databases evaluated'])
for j in groups:
    print('\nGrupo '+j)
    for i in rois:
        print('\nDatos de '+i)
        for k in bandas:
            if k in i:
                band=k
        for c in roi:
            if c in i:
                r=c
        if(datos[i].isnull().tolist()[0]==False):
            print('Normalidad')
            print(pg.normality(data=datos[datos['group'].isin([j])], dv=i, group='database',method='shapiro'))
            print('Varianza')
            print(pg.homoscedasticity(data=datos[datos['group'].isin([j])], dv=i, group='database',method='bartlett'))
            #EN general no estan dando con varianza igual
            print('Welch ANOVA')
            welch=pg.welch_anova(data=datos[datos['group'].isin([j])], dv=i, between='database')
            print(welch)
            valor=welch['p-unc'][0]
            if valor<0.05:
                print('Hay diferencias estadisticamente significativas entre las medias')
                print('No Comparables')
                result='group differences'
            else:
                print('No hay diferencias estadisticamente significativas entre las medias')
                print('Comparables')
                result='No group differences'

            print('Gameshowell test')
            a=pg.pairwise_gameshowell(data=datos[datos['group'].isin([j])], dv=i, between='database')
            print(a)
            databases=datos[datos['group'].isin(['Control'])]['database'].unique()
            b=a['A'].unique().tolist()+a['B'].unique().tolist()
            b=set(b)
            listToStr = ','.join([str(elem) for elem in b])
            nueva_fila = { 'Group':j,'ROI':r,'Band':band,'p-unc':valor,'Result':result,'Databases evaluated':listToStr} # creamos un diccionario
            resultados_comparacion = resultados_comparacion.append(nueva_fila, ignore_index=True)
resultados_comparacion.to_csv('Manipulacion- Rois-Componentes de todas las DB\Datosparaorganizardataframes\ResultadosWelchROIs.csv',sep=';',index=False)
# print(pg.welch_anova(data=datos, dv=i, between='database'))
# pg.pairwise_gameshowell(data=datos[datos['group']=='Control'], dv='C14_rDelta', between='database')

# pg.welch_anova(data=datos, dv='C14_rDelta', between='database') 

# Comparaciones múltiples
