import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend 'Agg' para no necesitar interfaz gráfica
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from training_functions import *
import joblib

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

        data1 = mapa_de_correlacion(data1, path_plot, var1)

        X1 = data1.values[:, :-1]  # La ultima posicion es el grupo, por eso se elimina
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
        feat1 = primeras_carateristicas(X_train1, sorted_names1, nombres_columnas1, features_scores1, feat1, index1, path_plot, var1)

        curva_de_aprendizaje(sorted_names1, data1, best_selected1, X_train1, y_train1, modelos1, acc_per_feature1, std_per_feature1, path_plot, var1)

        GS_fitted1 = best_selected1.fit(X_train1, y_train1)
        modelos1['GridSerach'] = GS_fitted1
        predicted1 = GS_fitted1.predict(X_test1)
        predicted_proba = GS_fitted1.predict_proba(X_test1)[:, 1]  # Probabilidades de la clase positiva
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
        best_features1 = sorted_names1[:pos_model1]
        mi_path1 = os.path.join(path_plot, 'best_params1.txt')
        with open(mi_path1, 'w') as f:
            for i in params1:
                f.write(f"{i}\n")
        
        title = f'validation_GridSearch.png'
        palette1 = ["#8AA6A3", "#127369"]

        curva_validacion3(GS_fitted1, X_train1, y_train1, title, palette1, var1)
        plt.grid()
        fig = plt.gcf()
        fig.savefig(path_plot + '/' + title, bbox_inches='tight')
        plt.close()

        acc1, std1, fbest_model1, input_best_index1 = features_best3(best_features1, best_selected1, data1.iloc[:, :-1], X_train1, y_train1, path_plot)

    else:
        # Ajuste fino del modelo existente
        data1 = mapa_de_correlacion(data1, path_plot, var1)

        X1 = data1.values[:, :-1]  # La ultima posicion es el grupo, por eso se elimina
        y1 = data1.values[:, -1]
        print(X1.shape)
        print(y1.shape)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X1,  # Valores de X para data1
            y1,  # Valores de Y para data1
            test_size=0.2,  # Test de 20%
            random_state=1,  # Semilla
            stratify=data1.values[:, -1])  # que se mantenga la proporcion en la división para data1

        # Ajuste del modelo existente
        model.fit(X_train1, y_train1)
        predicted1 = model.predict(X_test1)
        print(
            f"Classification report for classifier {model}:\n"
            f"{metrics.classification_report(y_test1, predicted1)}\n"
        )
        dataframe_metrics1 = metrics.classification_report(y_test1, predicted1, output_dict=True)
        dataframe_metrics1 = pd.DataFrame(dataframe_metrics1).T
        scores1 = cross_val_score(
            estimator=model,
            X=X_train1,
            y=y_train1,
            cv=10,
            n_jobs=-1
        )
        print('CV accuracy scores: %s' % scores1)
        print('\nCV accuracy: %.3f +/- %.3f' %
              (np.mean(scores1), np.std(scores1)))

        acc1 = np.mean(scores1)
        std1 = np.std(scores1)
        fbest_model1 = model
        input_best_index1 = None  
        var1 = str_ratio  

    return acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, var1, predicted_proba, X_test1, y_test1

def exec2(acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, var1, predicted_proba, X_test=None, y_test=None):               
    # Ruta para guardar los datos
    #output_file = os.path.join(path_plot, 'features_best3_data.pkl')

    #joblib.dump(fbest_model1, output_file)
    band = 0
    if X_test is None and y_test is None:
        X_test = X_train1
        y_test = y_train1
        band = 1

    print(acc1[-1])
    print(std1[-1])

    # Verifica los datos de entrada
    print("Datos de entrenamiento (X_train1):", X_train1.shape)
    print("Etiquetas de entrenamiento (y_train1):", np.unique(y_train1, return_counts=True))
    print("Datos de prueba (X_test1):", X_test.shape)
    print("Etiquetas de prueba (y_test1):", np.unique(y_test, return_counts=True))

    # Verifica el mapeo de las clases
    print("Mapeo de clases:", clases_mapeadas1)

    # Aplicar selección de características a X_test1
    X_test_selected = X_test[:, input_best_index1]

    if band == 1:
        predicted1 = fbest_model1.predict(X_test_selected)
        # Convertir las etiquetas de prueba y predicción utilizando el mapeo de clases
        y_test = np.array([clases_mapeadas1[label] for label in y_test])
    else:
        predicted1 = fbest_model1.predict(X_test_selected)

    classes_x1 = (predicted1 >= 0.5).astype(int)

    output_file1 = os.path.join(path_plot, f'{name}_result1.csv')  
    computerprecision(y_test, classes_x1, output_file1)

    # Calcula y muestra las métricas de precisión, recall y F1-score
    precision = precision_score(y_test, classes_x1)
    recall = recall_score(y_test, classes_x1)
    f1 = f1_score(y_test, classes_x1)
    auc2 = roc_auc_score(y_test, predicted_proba)  # AUC
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


    print(f'Precision: {precision}\nRecall: {recall}\nF1: {f1}\nAUC: {auc2}')

    # Guardar métricas en un archivo CSV
    metrics_dict = {
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'AUC': [auc2]
    }
    path_metrics_csv = os.path.join(path_plot, f'metrics_ML_{var1}.csv')
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(path_metrics_csv, index=False)

    dataframe_metrics2 = metrics.classification_report(y_test, predicted1, output_dict=True)
    dataframe_metrics2 = pd.DataFrame(dataframe_metrics2).T

    # Verifica las predicciones y etiquetas reales
    print("Predicciones:", predicted1)
    print("Etiquetas reales:", y_test)

    cm_test1 = confusion_matrix(y_test, classes_x1)
    plot_confusion_matrix(path_plot, var1, cm_test1, classes=class_names, title='Confusion matrix')


    title = 'validation_DecisionTree.png'
    palette1 = ["#8AA6A3","#127369"]

    if band == 1:
        curva_validacion3(fbest_model1, X_test_selected, y_train1, title, palette1, var1)
    else:
        curva_validacion3(fbest_model1, X_train1[:, input_best_index1], y_train1, title, palette1, var1)
    plt.grid()
    fig = plt.gcf()
    fig.savefig(path_plot + '/' + title, bbox_inches='tight')
    plt.close()

# Ejemplo de uso
neuros = ['neuroHarmonize']
names = ['G1']
space = 'ic'
ica = '58x25' #'54x10'
path = fr'/Users/imeag/Documents/udea/trabajoDeGrado/Data_analysis_ML_Harmonization_Proyect/Experimenting'
path_save = fr'/Users/imeag/Documents/udea/trabajoDeGrado/Data_analysis_ML_Harmonization_Proyect/Experimenting/Results'
#ratios = [79, 15, 31]
ratios = [79]
#class_names = ['HC', 'ACr']
class_names = ['ACr', 'HC']
# Inicializar fbest_model1 fuera del bucle
fbest_model1 = None

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
            
            path_plot = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{str_ratio}_{space}')
            file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
            data = pd.read_feather(file_path)
            data['group'] = data['group'].replace({'G1': 'ACr', 'Control': 'HC'})

            if str_ratio == '2to1':
                # Entrenamiento inicial con el dataset 2to1
                acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, var1,predicted_proba, X_test1, y_test1 = exec1(neuro, name, space, path_save, path_plot, data, str_ratio, class_names, model=None)
            else:
                # Ajuste fino con los datasets 5to1 y 10to1
                acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, var1,predicted_proba, X_test1, y_test1 = exec1(neuro, name, space, path_save, path_plot, data, str_ratio, class_names, model=fbest_model1)

            # Evaluación final del modelo
            fbest_model1 = exec2(acc1, std1, fbest_model1, input_best_index1, X_train1, y_train1, clases_mapeadas1, path_plot, var1, predicted_proba, X_test1, y_test1)
