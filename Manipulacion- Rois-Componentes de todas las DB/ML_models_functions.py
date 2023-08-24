#Importación librerías y carga de datos

"""
Modulo de entrenamiento para diferentes pipelines de ML para el
entrenamiento utilizando los datasets creados. El flujo es capaz de 
generalizar en la mayoría de los casos y únicamente es necesario
realizar modificaciones en la ruta a los datos. #cambiar a inglés
"""

import os
import pandas as pd 
import seaborn as sns                                                   
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, \
                                    cross_val_score, \
                                    learning_curve, \
                                    RandomizedSearchCV, \
                                    GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from boruta import BorutaPy
from tpot import TPOTClassifier
import dataframe_image as dfi
from keras.models import Sequential,load_model,model_from_json
import itertools
import joblib  
from sklearn.metrics import confusion_matrix, recall_score, precision_score

#----------------PRECISION AND RECALL----------------------------------------------------------------

def computerprecision(test_label,classes_x):
  '''
  Compute precision and recall
  '''

  precision_test = precision_score(test_label,classes_x)
  recall_test = recall_score(test_label, classes_x)
  f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
  print( 'Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )


#-----------------CONFUSION MATRIX---------------------------------------------------------------------
def confusion_matrix(path_plot,cm, classes,normalize=False,title='Confusion matrix',cmap='coolwarm'):

    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path_plot+'/'+'confusion_matrix.png')


#------------------------------------------DATA DEPURATION---------------------------------------------------------


def depuration(data,space,m,b=None,bm=None):

    '''Deletes unnecesary info, changes upper to lowercase and map string binary data.
    data = dataset
    space = the space can be 'roi' if its divided by regions of interest or 'ic' if its divided by components
    m=metrics / features
    b=drequency band
    roi[list] = the regions of interest if space = 'roi' or the components if space = 'ic'    
    bm=modulated crossed frequency bands e.g. ['Gamma']

    '''
    
    print('Initial data shape:')
    print(f'subjects: {data.shape[0]} | features: {data.shape[1]}')

    if space == 'roi':
        roi = ['F','C','T','PO'] #regions
    elif space == 'ic':
        roi = ['C14','C15','C18','C20','C22','C23','C24','C25'] #components
        
    #returns a modified dataframe also called data depending on m,b and roi
    if b!=None:

        for metrici in m:
            for bandi in b:
                for ri in roi:
                    if metrici != 'crossfreq':
                        data.drop([metrici+'_'+ri+'_'+bandi],axis=1,inplace=True)
                    else:
                        for modul in bm:
                            if modul == 'MGAMMA':
                                modul = modul[0]+modul[1:].swapcase()
                            else:
                                pass
                            try:
                                data.drop([metrici+'_'+ri+'_'+modul+'_'+bandi],axis=1,inplace=True)
                            except:
                                continue
    else:
        pass    
    
    #create a new dataframe with the columns to delete
    col_del = pd.DataFrame()

    #Delete columns with missing data (n/a)
    for column in data.columns:

        if data[column].isna().sum() != 0:
            col_del[column] = [data[column].isna().sum()]

            #print('{} : {}'.format(column, (data[column].isna().sum())))
            data.drop(column, axis=1, inplace=True)



    #Data mapping = converts string data into binary (0 or 1)

    #Group mapping -> Controls = 0 and patients (G1) = 1
    clases_mapeadas = {label:idx for idx,label  
                   in enumerate(np.unique(data['group']))}

    data.loc[:,'group'] = data.loc[:,'group'].map(clases_mapeadas)
    print('Mapped classes:') 
    print(clases_mapeadas)

    #Sex mapping -> F = 0 and M = 1
    sexo_mapeado = {label:idx for idx,label
                in enumerate(np.unique(data['sex']))}

    data.loc[:,'sex'] = data.loc[:,'sex'].map(sexo_mapeado)
    print('Mapped sex:') 
    print(sexo_mapeado)

    #Select numerical data only
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    data = data.select_dtypes(include=numerics)

    #Put 'group' column at the end
    target = data.pop('group')
    data.insert(len(data.columns), target.name, target)

    print('New data shape:')
    print(f'subjects: {data.shape[0]} | features: {data.shape[1]}')

    return data
    
     


#-------------------FEATURE SELECTION-----------------------------------------

#Train test split using train_test_split from sklearn.model_selection

#Heat map
def heat_map(data,save=False,path_plot=None):
    '''
    Returns a correlation heatmap of the data
    '''

    sns.heatmap(data.corr())
    plt.title('Correlation matrix for all features')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()
    if save==True:
        plt.savefig(path_plot+'/'+'correlation_all.png')

#-------CLASSIFICATION REPORT------------------------
def classif_report(X_train, y_train,X_test,y_test,estimator,print_report=True):

    "This function returns the cross validation scores of a given model and shows the classification report if print_report = True"

    scores = cross_val_score(estimator=estimator,
                        X=X_train,
                        y=y_train,
                        cv=10,
                        n_jobs=-1)
    
    predicted = estimator.predict(X_test)
    if print_report == True:
        print(
        f"Classification report for classifier {estimator}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    
    return scores

#---------------LEARNING CURVE-------------------------------------
def plot_learning_curve(acc_per_feature,std_per_feature,save=False,path_plot=None):

    "Plot the learning rate value for each feature"

    plt.plot(
        range(1, len(acc_per_feature)+1),
        acc_per_feature,
        color='red'
        ) 

    plt.fill_between(
                    range(1, len(acc_per_feature)+1),
                    np.array(acc_per_feature) + np.array(std_per_feature),
                    np.array(acc_per_feature) - np.array(std_per_feature),
                    alpha=0.15,
                    color='red'
                    )

    plt.grid()
    plt.title('Learning Curve Decision Tree')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')    
    plt.show()
    if save == True:
        plt.savefig(path_plot+'/'+'features_plot_all.png')

    



#------------RANDOM FOREST----------------------------------------

def random_forest(X_train, y_train):

    #Features definition as input for random forest model
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 30)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    criterion = ['gini',  'entropy', 'log_loss']

    #Creation of grid with features
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion
                }
    
    #Estimator definition
    forestclf_grid = RandomForestClassifier()

    #Model creation
    rf_random = RandomizedSearchCV(
                                estimator=forestclf_grid,
                                param_distributions=random_grid,
                                n_iter=100,
                                cv=10,
                                verbose=2,
                                random_state=10,
                                n_jobs=-1
                                )

    #Fit model to train data
    rf_random.fit(X_train, y_train)

    #Best selected model
    best_selected = rf_random.best_estimator_

    #Fit best selected model to train data
    GS_fitted = best_selected.fit(X_train, y_train)
    
    return GS_fitted, best_selected

#--------------TRAIN VS TEST GRAPH----------------------------------------------

def train_test_accuracy_plot(estimator, X_train, y_train, model_name, save=False, path_plot=None):
    """
    Generate a learning curve for the given estimator to visualize training and validation accuracy
    as a function of the number of training samples.

    Returns:
        None: The function generates and displays the learning curve plot.

    """
    
    # Generate learning curve data using the provided estimator, training features, and labels.
    train_sizes, train_scores, test_scores = \
        learning_curve(
            estimator=estimator,              # Estimator (classifier or regressor) to use for learning curve.
            X=X_train,                        # Training features.
            y=y_train,                        # Training labels.
            train_sizes=np.linspace(0.1, 1, 10),  # Array of floats representing the proportion of the dataset to use for learning curve points.
            cv=10,                            # Number of cross-validation folds.
            n_jobs=-1                         # Number of CPU cores to use for parallelization (-1 means use all available cores).
        )
    
    # Calculate the mean and standard deviation of training and testing scores across the cross-validation folds.
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the training accuracy with error bands.
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    # Plot the testing accuracy with error bands.
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    # Add plot labels and legend.
    plt.grid()
    plt.title(f'Validation curve for {model_name}')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.5])

    # Display the plot.
    plt.show()

    # Optionally, save the plot to a file if 'save' is True.
    if save:
        plt.savefig(path_plot + '/' + 'validation_GridSearch.png')


#-------------------DECISION TREE(BORUTA)--------------------------

def boruta(data,X_train,y_train,best_selected):


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

    #creates a new trainning dataset only with selected features
    X_transform = feat_selector.transform(X_train)

    #adjust the same model used in random forest to the newnew trainning dataset with the selected features
    boruta_fitted = best_selected.fit(X_transform, y_train) 

    return boruta_fitted, X_transform


#--------------FEATURE SELECTION WITH DECISION TREE---------------------------------------------------

def feat_selection_decision_tree(data,X_train,y_train,best_selected,plot=True,save=False,path_plot=None):

    '''Returns a list with the most important features and a bar plot with their relevance value if plot=True'''

    feat = pd.DataFrame()

    nombres_columnas = data.columns[:-1]
    best_selected.fit(X_train, y_train)
    features_scores = best_selected.feature_importances_
    #features_scores
    index = np.argsort(features_scores)[::-1]
    sorted_names = []

    for f in range(X_train.shape[1]):

        sorted_names.append(nombres_columnas[index[f]])
        print("%2d) %-*s %f" % (f + 1, 30,
                            nombres_columnas[index[f]],
                            features_scores[index[f]]))
        feat[nombres_columnas[index[f]]] = [features_scores[index[f]]]

        if plot==True:
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Relevance')

            plt.bar(range(X_train.shape[1]),
                features_scores[index],
                align='center')

            plt.xticks(range(X_train.shape[1]),
                nombres_columnas[index],
                rotation=90)

            plt.xlim([-1, 10])

            plt.tight_layout()
            plt.show()

            if save == True:
                plt.savefig(path_plot+'/'+'feature_importance.png')

    return sorted_names

        

#-----------------SVM (Grid search)---------------------------------------------------------

def svm_grid_search(X_train, y_train):
    svm_param_grid = {'C': list(np.logspace(-1, 4, 6)), 
                    'gamma': list(np.logspace(-3, 2, 6)) + ['Auto'] + ['scale'],
                    'kernel': ['rbf', 'poly']} 

    svc = SVC()
    svc_clf = GridSearchCV(
                        svc,
                        svm_param_grid,
                        n_jobs=-1,
                        cv=10
                        )

    svm_best_clf = svc_clf.fit(X_train, y_train)
    best_model = svm_best_clf.best_estimator_

    #Fit the best model to data
    svm_GS_fitted = best_model.fit(X_train, y_train)

    #Return the bes model fitted and the best selected model
    return svm_GS_fitted, best_model
    


#-------------- XGBoosting -----------------------------


def tpot(X,X_train, y_train):

    pipeline_optimizer = TPOTClassifier()

    pipeline_optimizer = TPOTClassifier(
                                    generations=5,
                                    population_size=int(X.shape[0]*0.4),
                                    cv=10,
                                    random_state=10,
                                    verbosity=3,
                                    n_jobs=-1
                                    )
    
    tpot_classifier = pipeline_optimizer.fit(X_train, y_train)

    return tpot_classifier










