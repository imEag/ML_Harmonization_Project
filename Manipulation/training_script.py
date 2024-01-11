import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from tpot import TPOTClassifier
from training_functions import *
from sklearn import datasets, metrics
import joblib
from sklearn.svm import SVC


def exec(neuro,name,space,path_save,data,var,class_names,model=None):
    # Directorio de resultados
    path_plot = os.path.join(path_save, f'graphics/ML/{neuro}/{name}_{var}_{space}')
    path_excel = os.path.join(path_save, f'tables/ML/{space}/{name}')
    path_excel1 = os.path.join(path_excel, f'describe_all_{var}.xlsx')
    path_excel2 = os.path.join(path_excel, f'describe_{var}.xlsx')
    path_excel3 = os.path.join(path_excel, f'features_{var}.xlsx')

    # Asegúrate de que la carpeta de destino exista
    os.makedirs(path_plot, exist_ok=True)
    if model is None:  
        # Código principal
        modelos = {}
        acc_per_feature = []
        std_per_feature = []
        print(f'sujetos: {data.shape[0]} | caracteristicas: {data.shape[1]}')

        # Preprocesamiento y análisis exploratorio de datos
        for group in data['group'].unique():
            print('{} : {}'.format(group, (data['group'] == group).sum()))

        # Asegúrate de que la carpeta de destino exista
        for path in [path_plot, path_excel]:
            os.makedirs(path, exist_ok=True)

        # Save Excel file
        data.describe().T.to_excel(path_excel1)
        data.groupby(by='group').describe().T.to_excel(path_excel2)

        # Selección de características y entrenamiento de modelos
        #m = ['power', 'sl', 'cohfreq', 'entropy', 'crossfreq']
        #bm = ['Mdelta', 'Mtheta', 'Malpha-1', 'Malpha-2', 'Mbeta1', 'Mbeta2', 'Mbeta3', 'Mgamma']

        #if space == 'roi':
        #    roi = ['F', 'C', 'T', 'PO']
        #elif space == 'ic':
        #    roi = ['C14', 'C15', 'C18', 'C20', 'C22', 'C23', 'C24', 'C25']

        #data = delcol(data, m, ['Gamma'], roi, bm)
        col_del = pd.DataFrame()

        # eliminación de columnas con datos faltantes
        for column in data.columns:
            if data[column].isna().sum() != 0:
                col_del[column] = [data[column].isna().sum()]
                print('{} : {}'.format(column, (data[column].isna().sum())))
                data.drop(column, axis=1, inplace=True)

        # Se mapean las clases
        clases_mapeadas = {label: idx for idx, label in enumerate(np.unique(data['group']))}
        data.loc[:,'group'] = data.loc[:,'group'].map(clases_mapeadas)
        print(clases_mapeadas)
        # Se elimina la columna, para ponerla al final
        target = data.pop('group')
        data.insert(len(data.columns), target.name, target)
        data['group'] = pd.to_numeric(data['group'])
        print(data.dtypes.unique())
        data.select_dtypes('O')
        data.groupby(by='sex').describe().T
        sexo_mapeado = {label: idx for idx, label in enumerate(np.unique(data['sex']))}
        data.loc[:,'sex'] = data.loc[:,'sex'].map(sexo_mapeado)
        print(sexo_mapeado)

        # data pasa a ser el arreglo únicamente con los datos númericos
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        data = data.select_dtypes(include=numerics)
        data.shape

        X = data.values[:,:-1]
        y = data.values[:,-1]
        print(X.shape)
        print(y.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            X, # Valores de X
            y, # Valores de Y
            test_size=0.2, # Test de 20%
            random_state=1, # Semilla
            stratify=data.values[:,-1]) # que se mantenga la proporcion en la división

        mapa_de_correlacion(data, path_plot)

        ### Árboles de decisión (Grid Search)
        random_grid = grid_search()
        rf_random = randomFo(random_grid,X_train, y_train)
        best_selected = rf_random.best_estimator_
        params=rf_random.best_params_
        # Guardar mejore carateristicas
        mi_path = path_plot+'/'+'best_params.txt'
        f = open(mi_path, 'w')

        for i in params:
            f.write(i+'\n')
        f.close()

        GS_fitted = best_selected.fit(X_train, y_train)
        modelos['GridSerach'] = GS_fitted
        predicted = GS_fitted.predict(X_test)
        print(
            f"Classification report for classifier {GS_fitted}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
            )
        dataframe_metrics = metrics.classification_report(y_test, predicted, output_dict=True)
        dataframe_metrics = pd.DataFrame(dataframe_metrics).T
        scores = cross_val_score(
                                estimator=GS_fitted,
                                X=X_train,
                                y=y_train,
                                cv=10,
                                n_jobs=-1
                                )
        print('CV accuracy scores: %s' % scores)
        print('\nCV accuracy: %.3f +/- %.3f' %
            (np.mean(scores), np.std(scores)))

        acc_per_feature.append(np.mean(scores))
        std_per_feature.append(np.std(scores))
        title = 'validation_GridSearch.png'
        curva_validacion(GS_fitted,X_train,y_train,path_plot,title)

        ## Arbles de decision (Boruta) #MODELO #1
        
        feat_selector = BorutaPy(
                                verbose=2,
                                estimator=best_selected,
                                max_iter=100,
                                random_state=10
                                )
        feat_selector.fit(X_train, y_train)
        selected_features = []
        print("\n------Support and Ranking for each feature------")
        for i in range(len(feat_selector.support_)):
            if feat_selector.support_[i]:
                print("Passes the test: ", data.columns[i],
                    " - Ranking: ", feat_selector.ranking_[i])
                selected_features.append(data.columns[i])

        # Guardar mejore carateristicas
        mi_path = path_plot+'/'+'best_features_boruta.txt'
        f = open(mi_path, 'w')

        for i in selected_features:
            f.write(i+'\n')
        f.close()

        X_transform = feat_selector.transform(X_train)
        boruta_fitted = best_selected.fit(X_transform, y_train)
        modelos['Boruta'] = boruta_fitted

        selected_features = [data.columns.get_loc(c) for c in selected_features if c in data]
        print(selected_features)

        predicted = boruta_fitted.predict(X_test[:,selected_features])

        print(
            f"Classification report for classifier {boruta_fitted}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

        dataframe_metrics = metrics.classification_report(y_test, predicted, output_dict=True)
        dataframe_metrics = pd.DataFrame(dataframe_metrics).T

        scores = cross_val_score(estimator=boruta_fitted,
                                X=X_transform,
                                y=y_train,
                                cv=10,
                                n_jobs=-1)

        print('CV accuracy scores: %s' % scores)

        print('\nCV accuracy: %.3f +/- %.3f' %
            (np.mean(scores), np.std(scores)))


        acc_per_feature.append(np.mean(scores))
        std_per_feature.append(np.std(scores))

        title = 'validation_Boruta.png'
        curva_validacion(boruta_fitted,X_transform,y_train,path_plot,title)
        
        classes_x=(predicted >= 0.5).astype(int)
        cm_test = confusion_matrix(y_test,classes_x)
        plot_confusion_matrix(path_plot,cm_test,classes=class_names,title='Confusion matrix')

        # Selección de caracteristicas con árboles de decisión MODELO #2
        feat = pd.DataFrame()
        nombres_columnas = data.columns[:-1]
        best_selected.fit(X_train, y_train)
        features_scores = best_selected.feature_importances_
        features_scores
        index = np.argsort(features_scores)[::-1]
        sorted_names = []

        feat = primeras_carateristicas(X_train, sorted_names,nombres_columnas,features_scores,feat,index,path_plot)

        #grafico_arbol_de_decision(best_selected,nombres_columnas)

        feat.describe().T.to_excel(path_excel3)

        curva_de_aprendizaje(sorted_names,data,best_selected,X_train,y_train,modelos,acc_per_feature,std_per_feature,path_plot)

        pos_model = np.argsort(acc_per_feature)[-1]
        best_model = list(modelos.keys())[pos_model]
        print(best_model)

        joblib.dump(modelos[best_model], path_plot+'/'+'modelo_entrenado.pkl') # Guardo el modelo.
        joblib.dump(best_selected, path_plot+'/'+'modelo_mejor.pkl') # Guardo el modelo.
        
        # Guardar mejore carateristicas
        best_features=sorted_names[:pos_model]
        mi_path = path_plot+'/'+'best_features.txt'
        f = open(mi_path, 'w')

        for i in best_features:
            f.write(i+'\n')
        f.close()
    else:
        best_selected = joblib.load(path_plot + '/modelo_mejor.pkl')
        best_features = pd.read_csv(path_plot+'/'+'best_features.txt', sep='\t') 

    new_data = pd.DataFrame()
    for i in range(0,len(data.columns)):
        for j in range(0,len(best_features)):
            if data.columns[i] == best_features[j]:
                new_data[best_features[j]] = data[best_features[j]]

    new_name = 'Data_complete_randomforest'+space+'_'+name
    new_data.reset_index(drop=True).to_feather(rf'{path_excel}\{new_name}.feather')

    mapa_de_correlacion(new_data, path_plot)

    selected_best_features = [new_data.columns.get_loc(c) for c in best_features if c in new_data]
    print(selected_best_features)
    acc, std, fbest_model, input_best_index = features_best(best_features,best_selected,data,X_train,y_train,path_plot)
    print(acc[-1])
    print(std[-1])
    predicted = fbest_model.predict(X_test[:,input_best_index])
    classes_x=(predicted >= 0.5).astype(int)
    output_file = path_plot + '/' + new_name + '_.csv'
    computerprecision(y_test, classes_x, output_file)

    cm_test = confusion_matrix(y_test,classes_x)
    plot_confusion_matrix(path_plot,cm_test,classes=class_names,title='Confusion matrix')

    title = 'validation_1_DecisionTree.png'
    curva_validacion(fbest_model,X_train[:, input_best_index],y_train,path_plot,title)

    #best_best_features=best_features[:15]
    #acc, std, fbest_model, input_best_index = features_best(best_best_features,best_selected,data,X_train,y_train,path_plot)
    #print(acc[-1])
    #print(std[-1])
    #predicted = fbest_model.predict(X_test[:,input_best_index])
    #classes_x=(predicted >= 0.5).astype(int)
    #output_file = path_plot + '/' + title[:-4] + '_.csv'
    #computerprecision(y_test, classes_x, output_file)

    #cm_test = confusion_matrix(y_test,classes_x)
    #plot_confusion_matrix(path_plot,cm_test,classes=class_names,title='Confusion matrix')

    #title = 'validation_DecisionTree.png'
    #curva_validacion(fbest_model,X_transform,y_train,path_plot,title)


#neuro = 'neuroHarmonize'
#name = 'G1'
#space = 'ic'
#path_save = r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\Data_analysis_ML_Harmonization_Proyect\Manipulacion- Rois-Componentes de todas las DB\Resultados'
#data = pd.read_feather(r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\Data_analysis_ML_Harmonization_Proyect\Manipulacion- Rois-Componentes de todas las DB\Dataframes\Data_complete_ic_neuroHarmonize_G1.feather')
#var = ''
#class_names=['Control','G1']
#exec(neuro,name,space,path_save,data,var,class_names)


neuro = 'neuroHarmonize'
name = 'G1'
space = 'ic'
path_save = r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\Data_analysis_ML_Harmonization_Proyect\Manipulacion- Rois-Componentes de todas las DB\Resultados'
data = pd.read_feather(r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\Data_analysis_ML_Harmonization_Proyect\Manipulacion- Rois-Componentes de todas las DB\Dataframes\Data_complete_ic_neuroHarmonize_G1_54x10.feather')
var = '54x10'
class_names=['Control','G1']
exec(neuro,name,space,path_save,data,var,class_names)


neuro = 'sovaharmony'
name = 'G1'
space = 'ic'
path_save = r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\Data_analysis_ML_Harmonization_Proyect\Manipulacion- Rois-Componentes de todas las DB\Resultados'
data = pd.read_feather(r'E:\Academico\Universidad\Posgrado\Tesis\Paquetes\Data_analysis_ML_Harmonization_Proyect\Manipulacion- Rois-Componentes de todas las DB\Dataframes\Data_complete_ic_sovaharmony_G1.feather')
var = ''
class_names=['Control','G1']
exec(neuro,name,space,path_save,data,var,class_names)


