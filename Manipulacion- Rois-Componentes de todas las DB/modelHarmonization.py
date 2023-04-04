# Importación de librerías necesarias

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

def select_metric(data_roi,data_com,metric):
    # Se extraen las potencias  en ambos datasets
    # (filtrando columnas que contengan la palabra 'metric' en su nombre).

    metric_data_roi = data_roi.filter(regex=metric)

    # Se añade la columna de grupo
    # la cual corresponde al target en la clasificación

    metric_data_roi.loc[:,'group'] = data_roi.loc[:,'group']

    # Mismo procedimiento para el df de componentes

    metric_data_com = data_com.filter(regex=metric)
    metric_data_com.loc[:,'group'] = data_com.loc[:,'group']

    return metric_data_roi,metric_data_com 


def validation_metric(metric_data_roi,metric_data_com):
    # Verificación de que las etiquetas hayan quedado en la posición de que debe
    # ser utilizando la igualdad elemento a elemento.
    print((metric_data_roi['group'] != data_roi['group']).sum())
    print((metric_data_com['group'] != data_com['group']).sum())

def maps(metric_data_roi,metric_data_com):
    # Se realiza un mapeo de clases donde se asigna una etiqueta númerica 
    # a cada clase (desde 0 hasta 3)

    clases_mapeadas = {label:idx for idx,label in enumerate(np.unique(metric_data_roi['group']))}
    print(clases_mapeadas)

    # Por verificación, se hace igualmente para el dataset de componentes

    clases_mapeadas = {label:idx for idx,label in enumerate(np.unique(metric_data_com['group']))}
    print(clases_mapeadas)

    # Se realiza el mapeo de las clases 

    metric_data_roi.loc[:,'group'] = metric_data_roi.loc[:,'group'].map(clases_mapeadas) 
    metric_data_com.loc[:,'group'] = metric_data_com.loc[:,'group'].map(clases_mapeadas)

    print(metric_data_roi['group'].unique())
    print(metric_data_com['group'].unique())

def tree(metric_data,names_cols,n_estimators,criterion,random_state):
    ## Identificación de caracteristicas de relevancia a través de árboles de decisión 

    
    X_train, X_test, y_train, y_test = train_test_split(
        metric_data.values[:,:-1],
        metric_data.values[:,-1],
        test_size=0.2,
        random_state=1,
        stratify=metric_data_roi.values[:,-1])
    forestclf = RandomForestClassifier(n_estimators=500,
        random_state=1)
    forestclf.fit(X_train, y_train)
    features_scores = forestclf.feature_importances_
    features_scores
    index = np.argsort(features_scores)[::-1]


    for f in range(X_train.shape[1]):

        print("%2d) %-*s %f" % (f + 1, 30,
                            names_cols[index[f]],
                            features_scores[index[f]]))

    plt.title('Importancia de la caracteristica')

    plt.bar(range(X_train.shape[1]),
        features_scores[index],
        align='center')

    plt.xticks(range(X_train.shape[1]),
        names_cols[index],
        rotation=90)

    plt.xlim([-1, X_train.shape[1]])

    plt.tight_layout()

    plt.show()

    return X_train, X_test, y_train, y_test

def svc(X_train, y_train,X_test,y_test):
    supvm = SVC(C=10, random_state=1)
    supvm.fit(X_train, y_train)
    predicted = supvm.predict(X_test)
    print(f"Classification report for classifier {supvm}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    supvm.fit(X_train, y_train)
    predicted = supvm.predict(X_test)
    print(f"Classification report for classifier {supvm}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

def fig(data):
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    sns.scatterplot(data=data,
                y='power_C14_Delta',
                x='power_C14_Theta',
                hue='group',
                style='database',
                )
    metric_data = data.filter(regex='power')
    metric_data['group'] = data['group']
    metric_data

def GaussianNB_Harmonize(X_train, y_train,X_test,y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n")
    
def tpotHarmonize(X_train, y_train,X_test,y_test,name_cols):
    pipeline_optimizer = TPOTClassifier()
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=100, cv=5,
                                    random_state=1, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    forestclf = RandomForestClassifier(bootstrap=False,
                            criterion='entropy',
                            max_features=0.05,
                            min_samples_leaf=5, 
                            min_samples_split=13,
                              n_estimators=100)
    forestclf.fit(X_train, y_train)
    features_scores = forestclf.feature_importances_
    features_scores
    index = np.argsort(features_scores)[::-1]


    for f in range(X_train.shape[1]):

        print("%2d) %-*s %f" % (f + 1, 30,
                            name_cols[index[f]],
                            features_scores[index[f]]))

    plt.title('Importancia de la caracteristica')

    plt.bar(range(X_train.shape[1]),
        features_scores[index],
        align='center')

    plt.xticks(range(X_train.shape[1]),
        name_cols[index],
        rotation=90)

    plt.xlim([-1, 10])

    plt.tight_layout()

    plt.show()
    
# Rutas donde cada persona tiene los df locales, de la carpeta de OneDrive
# Copiar la ruta como una variable y cambiar la varibale en el siguiente
# bloque de código.

path_santiago = r'C:\Users\santi\Universidad de Antioquia\VALERIA CADAVID CASTRO - Resultados_Armonizacion_BD'
path_veronica = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD'
path = path_veronica

# IC 
print('Data_complete_ic_sovaharmony_G2G1\n','Data_complete_ic_neuroHarmonize_G1G2')
data_ic_sova_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_G2G1.feather'.format(path=path))
data_ic_harmo_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_G2G1.feather'.format(path=path))

print('Data_complete_ic_sovaharmony_G1\n','Data_complete_ic_neuroHarmonize_G1')
data_ic_sova_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_G1.feather'.format(path=path))
data_ic_harmo_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_G1.feather'.format(path=path))

print('Data_complete_ic_sovaharmony_DTA\n','Data_complete_ic_neuroHarmonize_DTA')
data_ic_sova_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_DTA.feather'.format(path=path))
data_ic_harmo_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_DTA.feather'.format(path=path))

#data_ic_sova_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_sovaharmony_DCL.feather'.format(path=path))
#data_ic_harmo_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_ic_neuroHarmonize_DCL.feather'.format(path=path))

# ROI
print('Data_complete_roi_sovaharmony_G2G1\n','Data_complete_roi_neuroHarmonize_G2G1')
data_roi_sova_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_G2G1.feather'.format(path=path))
data_roi_harmo_G1G2=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_G2G1.feather'.format(path=path))

print('Data_complete_roi_sovaharmony_G1\n','Data_complete_roi_neuroHarmonize_G1')
data_roi_sova_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_G1.feather'.format(path=path))
data_roi_harmo_CTR=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_G1.feather'.format(path=path))

print('Data_complete_roi_sovaharmony_DTA\n','Data_complete_roi_neuroHarmonize_DTA')
data_roi_sova_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_DTA.feather'.format(path=path))
data_roi_harmo_DTA=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_DTA.feather'.format(path=path))

#data_roi_sova_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_sovaharmony_DCL.feather'.format(path=path))
#data_roi_harmo_DCL=pd.read_feather(r'{path}\Datosparaorganizardataframes\Data_complete_roi_neuroHarmonize_DCL.feather'.format(path=path))

data_roi = data_ic_sova_G1G2
data_com = data_ic_harmo_G1G2
print('Shape ROI',data_roi.shape)
print('Shape Components',data_com.shape)

statistics_roi = data_roi.describe()
statistics_com = data_com.describe()

#metric_data_roi,metric_data_com = select_metric(data_roi,data_com,'power')
#
#statistics_roi_pwr = metric_data_roi.describe()
#statistics_com_pwr = metric_data_com.describe()

validation_metric(data_roi,data_com)
maps(data_roi,data_com)

names_cols = data_roi.columns[:-1]
print("ROI")
X_train_roi, X_test_roi, y_train_roi, y_test_roi=tree(data_roi,names_cols,n_estimators=100,criterion='gini',random_state=1)

print("COMPONENTS")
X_train_comp, X_test_comp, y_train_comp, y_test_comp=tree(data_com,names_cols,n_estimators=100,criterion='gini',random_state=1)

svc(X_train_roi, y_train_roi,X_test_roi,y_test_roi)
svc(X_train_comp, y_train_comp,X_test_comp,y_test_comp)

fig(data_roi)
fig(data_com)

tpotHarmonize(X_train_roi, y_train_roi,X_test_roi,y_test_roi,names_cols)
tpotHarmonize(X_train_comp, y_train_comp,X_test_comp,y_test_comp,names_cols)
GaussianNB_Harmonize(X_train_roi, y_train_roi,X_test_roi,y_test_roi)
GaussianNB_Harmonize(X_train_comp, y_train_comp,X_test_comp,y_test_comp)



