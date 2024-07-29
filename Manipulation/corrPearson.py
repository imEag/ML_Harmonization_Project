# In[]
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def start(data1, var1,path_save,space,name):
    path_excel1 = os.path.join(path_save, f'')
    path_excel1_1 = os.path.join(path_excel1, f'describe_all_{var1}.xlsx')
    path_excel1_2 = os.path.join(path_excel1, f'describe_{var1}.xlsx')
    # Save Excel file for data1
    data1.describe().T.to_excel(path_excel1_1)
    data1.groupby(by='group').describe().T.to_excel(path_excel1_2)
    col_del1 = pd.DataFrame()

    # Eliminación de columnas con datos faltantes para data1
    for column in data1.columns:
        if data1[column].isna().sum() != 0:
            col_del1[column] = [data1[column].isna().sum()]
            print('{} : {}'.format(column, (data1[column].isna().sum())))
            data1.drop(column, axis=1, inplace=True)

    # Se mapean las clases para data1
    clases_mapeadas1 = {label: idx for idx, label in enumerate(np.unique(data1['group']))}
    data1.loc[:, 'group'] = data1.loc[:, 'group'].map(clases_mapeadas1)
    print(clases_mapeadas1)

    # Se elimina la columna, para ponerla al final para data1
    target1 = data1.pop('group')
    data1.insert(len(data1.columns), target1.name, target1)
    data1['group'] = pd.to_numeric(data1['group'])
    print(data1.dtypes.unique())
    data1.select_dtypes('O')
    data1.groupby(by='sex').describe().T
    sexo_mapeado1 = {label: idx for idx, label in enumerate(np.unique(data1['sex']))}
    data1.loc[:, 'sex'] = data1.loc[:, 'sex'].map(sexo_mapeado1)
    print(sexo_mapeado1)

    # data1 pasa a ser el arreglo únicamente con los datos númericos
    numerics1 = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data1 = data1.select_dtypes(include=numerics1)
    data1.shape
    aCorr(data1, path_excel1)


def aCorr(data, path_excel1):
    # Calcular la matriz de correlación de Pearson
    correlation_matrix = data.corr()
    save_corr_excel(correlation_matrix, path_excel1)
    graph_corr(data, correlation_matrix, path_excel1)


def save_corr_excel(correlation_matrix, path_excel1):
    # Configurar el umbral de correlación
    threshold = 0.8

    # Listas para almacenar características altamente correlacionadas y eliminadas
    highly_correlated_features = []
    features_to_drop = []

    # Encontrar pares de características con correlación mayor al umbral
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                highly_correlated_features.append((colname_i, colname_j))
                if colname_i not in features_to_drop:
                    features_to_drop.append(colname_i)

    # Crear DataFrames para las listas
    correlated_df = pd.DataFrame(highly_correlated_features, columns=['Feature 1', 'Feature 2'])
    dropped_df = pd.DataFrame(features_to_drop, columns=['Dropped Features'])

    # Guardar las listas en un archivo Excel
    with pd.ExcelWriter(os.path.join(path_excel1, 'correlation_report.xlsx')) as writer:
        correlated_df.to_excel(writer, sheet_name='Highly Correlated', index=False)
        dropped_df.to_excel(writer, sheet_name='Dropped Features', index=False)
        correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')

    print("Reporte guardado en 'correlation_report.xlsx'")


def graph_corr(data, correlation_matrix, path_excel1):
    # Generar la gráfica de correlación antes de eliminar características
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Mapa de calor de correlación antes de eliminar características")
    plt.savefig(os.path.join(path_excel1, 'correlation_before.png'))
    # plt.show()

    # Configurar el umbral de correlación
    threshold = 0.8

    # Listas para almacenar características altamente correlacionadas y eliminadas
    highly_correlated_features = []
    features_to_drop = []

    # Encontrar pares de características con correlación mayor al umbral
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname_i = correlation_matrix.columns[i]
                colname_j = correlation_matrix.columns[j]
                highly_correlated_features.append((colname_i, colname_j))
                if colname_i not in features_to_drop:
                    features_to_drop.append(colname_i)

    # Eliminar las características altamente correlacionadas
    data_reduced = data.drop(columns=features_to_drop)

    # Calcular la nueva matriz de correlación
    new_correlation_matrix = data_reduced.corr()

    # Generar la gráfica de correlación después de eliminar características
    plt.figure(figsize=(10, 8))
    sns.heatmap(new_correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Mapa de calor de correlación después de eliminar características")
    plt.savefig(os.path.join(path_excel1, 'correlation_after.png'))
    # plt.show()

    # Guardar las listas en un archivo Excel
    correlated_df = pd.DataFrame(highly_correlated_features, columns=['Feature 1', 'Feature 2'])
    dropped_df = pd.DataFrame(features_to_drop, columns=['Dropped Features'])

     # Guardar el DataFrame reducido como archivo Feather
    feather_file_path = os.path.join(path_excel1, f'Data_integration_{space}_{neuro}_{name}.feather')
    data_reduced.reset_index(drop=True).to_feather(feather_file_path)

    #with pd.ExcelWriter(os.path.join(path_excel1, 'correlation_report.xlsx'), mode='a') as writer:
    #    correlated_df.to_excel(writer, sheet_name='Highly Correlated', index=False)
    #    dropped_df.to_excel(writer, sheet_name='Dropped Features', index=False)
    #    correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix Before')
    #    new_correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix After')

    print(f"Reporte guardado en 'correlation_report.xlsx' y características finales guardadas en '{feather_file_path}'")

# Ejemplo de uso
neuros = ['neuroHarmonize', 'sovaharmony']
names = ['G1']
space = 'ic'
path = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper_V2\dataframes"
path_save = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Eli"
ratios = [79, 31, 15]
class_names = ['Control', 'ACr']

for neuro in neuros:
    for name in names:
        for ratio in ratios:
            if ratio == 79:
                str_ratio = '2to1'
            elif ratio == 31:
                str_ratio = '5to1'
            elif ratio == 15:
                str_ratio = '10to1'
            else:
                str_ratio = str(ratio)
            file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
            data = pd.read_feather(file_path)
            start(data, str_ratio,path_save,space,name)