"""
Function exec1
This function takes as input neuro, name, space, path_save, path_plot, data1, var1, class_names, and model.
It performs the following tasks:
Data preprocessing: removal of columns with missing data, class mapping, selection of numeric features.
Exploratory data analysis: calculation of descriptive statistics, visualization of correlations.
Model training: grid search for hyperparameter selection, training of model with best hyperparameters.
Evaluation of metrics: calculation of accuracy, precision, recall, F1-score.
Saving of results: saving of metrics, trained model, and selected features.
Function exec2
This function takes as input acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, and var1.
It performs the following tasks:
Data preprocessing: removal of columns with missing data, class mapping, selection of numeric features.
Model training: training of model with best hyperparameters obtained in exec1.
Evaluation of metrics: calculation of precision, recall, F1-score.
Saving of results: saving of metrics and confusion matrix.
Example usage
The code iterates over different combinations of neuro, name, and ratio.
For each combination, it calls exec1 to preprocess and train a model on data1.
Then, it calls exec2 to train the model on data2 using the features selected in exec1.
Notes
The code uses libraries such as pandas, matplotlib, scikit-learn, and joblib.
The code uses preprocessing techniques such as removal of columns with missing data, class mapping, and selection of numeric features.
The code uses feature selection techniques such as grid search and feature selection based on feature importance.
The code uses metrics such as accuracy, precision, recall, and F1-score to evaluate the performance of the models.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend 'Agg' para no necesitar interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from tpot import TPOTClassifier
from training_functions import *
from sklearn import metrics
import joblib
from sklearn.svm import SVC
import pickle

def exec1(neuro, name, space, path_save, path_plot, data1, var1, class_names, model=None):
    
    # Directorio de resultados para data1
    path_excel1 = os.path.join(path_save, f'tables/ML/{space}/{name}')
    path_excel1_1 = os.path.join(path_excel1, f'describe_all_{var1}.xlsx')
    path_excel1_2 = os.path.join(path_excel1, f'describe_{var1}.xlsx')
    path_excel1_3 = os.path.join(path_excel1, f'features_{var1}.xlsx')

    # Asegúrate de que la carpeta de destino exista
    os.makedirs(path_plot, exist_ok=True)

    if model is None:
        # Código principal
        modelos1 = {}
        acc_per_feature1 = []
        std_per_feature1 = []
        print(f'sujetos (data1): {data1.shape[0]} | características (data1): {data1.shape[1]}')

        # Preprocesamiento y análisis exploratorio de datos para data1
        for group in data1['group'].unique():
            print('{} : {}'.format(group, (data1['group'] == group).sum()))

        # Asegúrate de que la carpeta de destino exista para data1
        for path in [path_plot, path_excel1]:
            os.makedirs(path, exist_ok=True)

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

        data1 = mapa_de_correlacion(data1, path_plot,var1)

        X1 = data1.values[:, :-1] #La ultima posicion es el grupo, por eso se elimina
        y1 = data1.values[:, -1]
        print(X1.shape)
        print(y1.shape)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X1,  # Valores de X para data1
            y1,  # Valores de Y para data1
            test_size=0.2,  # Test de 20%
            random_state=1,  # Semilla
            stratify=data1.values[:, -1])  # que se mantenga la proporcion en la división para data1
        
        random_grid1 = grid_search()
        rf_random1 = randomFo(random_grid1, X_train1, y_train1)
        best_selected1 = rf_random1.best_estimator_
        params1 = rf_random1.best_params_

        # Guardar mejores características
        feat1 = pd.DataFrame()
        sorted_names1 = []
        nombres_columnas1 = data1.columns[:-1]
        features_scores1 = best_selected1.feature_importances_
        index1 = np.argsort(features_scores1)[::-1]
        feat1 = primeras_carateristicas(X_train1, sorted_names1, nombres_columnas1, features_scores1, feat1, index1, path_plot,var1)

        curva_de_aprendizaje(sorted_names1, data1, best_selected1, X_train1, y_train1, modelos1, acc_per_feature1, std_per_feature1, path_plot,var1)

        GS_fitted1 = best_selected1.fit(X_train1, y_train1)
        modelos1['GridSerach'] = GS_fitted1
        predicted1 = GS_fitted1.predict(X_test1)
        print(
            f"Classification report for classifier {GS_fitted1}:\n"
            f"{metrics.classification_report(y_test1, predicted1)}\n"
        )
        dataframe_metrics1 = metrics.classification_report(y_test1, predicted1, output_dict=True)
        dataframe_metrics1 = pd.DataFrame(dataframe_metrics1).T
        scores1 = cross_val_score(
            estimator=GS_fitted1,
            X=X_train1,
            y=y_train1,
            cv=10,
            n_jobs=-1
        )
        print('CV accuracy scores: %s' % scores1)
        print('\nCV accuracy: %.3f +/- %.3f' %
              (np.mean(scores1), np.std(scores1)))

        acc_per_feature1.append(np.mean(scores1))
        std_per_feature1.append(np.std(scores1))

        pos_model1 = np.argsort(acc_per_feature1)[-1]
        best_model1 = list(modelos1.keys())[pos_model1]
        best_features1=sorted_names1[:pos_model1]
        mi_path1 = path_plot+'/'+'best_params1.txt'
        f = open(mi_path1, 'w')

        for i in params1:
            f.write(i+'\n')
        f.close()       
        
        title = f'validation_GridSearch.png'
        palette1 = ["#8AA6A3","#127369"]

        curva_validacion3(GS_fitted1, X_train1,y_train1,title,palette1,var1)
        plt.grid()
        fig = plt.gcf()
        fig.savefig(path_plot + '/' + title, bbox_inches='tight')
        plt.close()

        acc1, std1, fbest_model1, input_best_index1 = features_best3(best_features1,best_selected1,data1.iloc[:, :-1],X_train1,y_train1,path_plot)

    return acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, var1, X_test1, y_test1


def exec2(neuro, name, space, path_save, data2, var2, class_names, acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, var1):
    
    # Directorio de resultados para data2
    path_excel2 = os.path.join(path_save, f'tables/ML/{space}/{name}')
    path_excel2_1 = os.path.join(path_excel2, f'describe_all_{var2}.xlsx')
    path_excel2_2 = os.path.join(path_excel2, f'describe_{var2}.xlsx')
    path_excel2_3 = os.path.join(path_excel2, f'features_{var2}.xlsx')

    # Asegúrate de que la carpeta de destino exista
    os.makedirs(path_plot, exist_ok=True)
    print(f'sujetos (data2): {data2.shape[0]} | características (data2): {data2.shape[1]}')
    for group in data2['group'].unique():
        print('{} : {}'.format(group, (data2['group'] == group).sum()))

    # Asegúrate de que la carpeta de destino exista para data2
    for path in [path_plot, path_excel2]:
        os.makedirs(path, exist_ok=True)

    # Save Excel file for data2
    data2.describe().T.to_excel(path_excel2_1)
    data2.groupby(by='group').describe().T.to_excel(path_excel2_2)
    col_del2 = pd.DataFrame()

    # Eliminación de columnas con datos faltantes para data2
    for column in data2.columns:
        if data2[column].isna().sum() != 0:
            col_del2[column] = [data2[column].isna().sum()]
            print('{} : {}'.format(column, (data2[column].isna().sum())))
            data2.drop(column, axis=1, inplace=True)

    # Se mapean las clases para data2
    clases_mapeadas2 = {label: idx for idx, label in enumerate(np.unique(data2['group']))}
    data2.loc[:, 'group'] = data2.loc[:, 'group'].map(clases_mapeadas2)
    print(clases_mapeadas2)

    # Se elimina la columna, para ponerla al final para data2
    target2 = data2.pop('group')
    data2.insert(len(data2.columns), target2.name, target2)
    data2['group'] = pd.to_numeric(data2['group'])
    print(data2.dtypes.unique())
    data2.select_dtypes('O')
    data2.groupby(by='sex').describe().T
    sexo_mapeado2 = {label: idx for idx, label in enumerate(np.unique(data2['sex']))}
    data2.loc[:, 'sex'] = data2.loc[:, 'sex'].map(sexo_mapeado2)
    print(sexo_mapeado2)

    # data2 pasa a ser el arreglo únicamente con los datos númericos
    numerics2 = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data2 = data2.select_dtypes(include=numerics2)
    data2.shape

    data2 = mapa_de_correlacion(data2, path_plot, var2)

    X2 = data2.values[:, :-1]  # La ultima posicion es el grupo, por eso se elimina
    y2 = data2.values[:, -1]
    print(X2.shape)
    print(y2.shape)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2,  # Valores de X para data2
        y2,  # Valores de Y para data2
        test_size=0.2,  # Test de 20%
        random_state=1,  # Semilla
        stratify=data2.values[:, -1])  # que se mantenga la proporcion en la división para data2

    input1 = np.array(data2.columns[input_best_index1])
    print(f'input_best_index1 {input_best_index1}')
    print(f'data2.columns[input_best_index1] {input1}')
    print(f'Best model1: {fbest_model1}')

    title = f'validation_final_{var2}.png'
    palette2 = ["#8AA6A3","#127369"]

    curva_validacion3(fbest_model1, X_train2, y_train2, title, palette2, var2)
    plt.grid()
    fig = plt.gcf()
    fig.savefig(path_plot + '/' + title, bbox_inches='tight')
    plt.close()

    # Calculo de métricas finales y matrices de confusión
    GS_fitted2 = fbest_model1.fit(X_train2, y_train2)
    predicted2 = GS_fitted2.predict(X_test2)
    cm2 = confusion_matrix(y_test2, predicted2)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm2, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax.text(x=j, y=i, s=cm2[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.grid()
    fig.savefig(path_plot + '/' + f'confusion_matrix_{var2}.png', bbox_inches='tight')
    plt.close()

    precision2 = precision_score(y_test2, predicted2, average='weighted')
    recall2 = recall_score(y_test2, predicted2, average='weighted')
    f1_2 = f1_score(y_test2, predicted2, average='weighted')
    print(f'Precision: {precision2}\nRecall: {recall2}\nF1: {f1_2}')
    dataframe_metrics2 = metrics.classification_report(y_test2, predicted2, output_dict=True)
    dataframe_metrics2 = pd.DataFrame(dataframe_metrics2).T

    return precision2, recall2, f1_2, dataframe_metrics2, clases_mapeadas2, path_plot, var2

        
# Ejemplo de uso
neuros = ['neuroHarmonize']
names = ['G1']
space = 'ic'
path = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper\dataframes"
path_save = r"C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_Paper\Resultados"
ratios = [79, 31, 15]
class_names = ['ACr', 'HC']

for neuro in neuros:
    for name in names:
        for ratio in ratios:
            if ratio == 79:
                str_ratio = '2to1'
                path_plot = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{str_ratio}_{space}')
                file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
                data = pd.read_feather(file_path)
                data['group'] = data['group'].replace({'G1': 'ACr', 'Control': 'HC'})
                acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, var1, X_test1, y_test1 = exec1(
                                    neuro, name, space, path_save, path_plot, data, str_ratio, class_names, model=None)
                
                # Verificación del índice antes de llamar a exec2
                max_index = len(data.columns) - 1
                valid_input_best_index1 = [i for i in input_best_index1 if i <= max_index]

                if not valid_input_best_index1:
                    raise IndexError("input_best_index1 contiene índices fuera del rango de las columnas de data")

                exec2(acc1, std1, fbest_model1, valid_input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, var1)
            
            else:
                x = data.select_dtypes(include=[np.number])
                y = data['group'].values
                
                # En los casos de ratio 31 y 15, usamos las variables definidas en el bloque ratio 79
                if ratio == 31:
                    str_ratio = '5to1'
                    path_plot = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{str_ratio}_{space}')
                    file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
                    data = pd.read_feather(file_path)
                    data['group'] = data['group'].replace({'G1': 'ACr', 'Control': 'HC'})
                elif ratio == 15:
                    str_ratio = '10to1'
                    path_plot = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{str_ratio}_{space}')
                    file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
                    data = pd.read_feather(file_path)
                    data['group'] = data['group'].replace({'G1': 'ACr', 'Control': 'HC'})

                # Usamos las variables de entrada y salida ya obtenidas
                exec2(acc1, std1, fbest_model1, valid_input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, var1)

    