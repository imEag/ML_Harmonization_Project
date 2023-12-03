#The functions created in the _funciones file are imported
#from ML_models_functions import computerprecision,confusion_matrix,depuration,heat_map, classif_report, random_forest,train_test_accuracy_plot, boruta, feat_selection_decision_tree, plot_learning_curve, svm_grid_search, tpot
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
tk.Tk().withdraw() # part of the import if you are not using other tkinter functions

#path_save = askdirectory()
path_save = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados_Eli'
print("user chose", path_save, "for save")
neuro = 'sovaHarmony' #database
name = 'G1' #group
space = 'ic' #space
var = ''

path_save = os.path.join(path_save,'Resultados')
path_plot = path_save +rf'\graphics\ML/{neuro}/{name}_{var}_{space}'

#path = askdirectory()
path = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\Resultados_Armonizacion_BD\Datosparaorganizardataframes'
print("user chose", path_save, "for read feather")
path_df = rf'{path}\Data_complete_{space}_{neuro}_{name}.feather'

data = pd.read_feather(path_df)

#Dictionary with the models best predicted values 
modelos = {}

#-------DATA DEPURATION---------------------------------------------------------------


m = ['power','sl','cohfreq','entropy','crossfreq'] #metrics / features
b=['Gamma']
bm = ['Mdelta','Mtheta','Malpha-1','Malpha-2','Mbeta1','Mbeta2','Mbeta3','Mgamma'] #modulated crossed frequency bands

data = depuration(data,space,m) #new dataset


#----------TRAIN TEST SPLIT-------------------------------
X = data.values[:,:-1] #all the rows except the last one
y = data.values[:,-1] #all the columns except the last one

X_train, X_test, y_train, y_test = train_test_split(                            
X, # Valores de X
y, # Valores de Y
test_size=0.2, # Test de 20%
random_state=1, # Semilla
stratify=data.values[:,-1]) # que se mantenga la proporcion en la división

#-------HEAT MAP------------------
#heat_map(data)

#------------RANDOM FOREST -----------------------------------------

#Grid search fitted predict
GS_fitted,best_selected = random_forest(X_train, y_train)
'''print(GS_fitted)
 
#Add to the models dictionary
modelos['GridSerach'] = GS_fitted

#Print the classification report using the function
classif_report(X_train, y_train,X_test,y_test,GS_fitted)'''
predicted = GS_fitted.predict(X_test)
print(predicted)
#Train vs test accuracy plot
train_test_accuracy_plot(GS_fitted,X_train,y_train,model_name='Grid search')


#--------------BORUTA------------------------------------
boruta_fitted, X_transform,best_selected = boruta(data,X_train,y_train,best_selected)
#Add to the models dictionary
modelos['Boruta'] = boruta_fitted

#Classification report
classif_report(X_transform, y_train,X_test,y_test,boruta_fitted,best_selected=best_selected)

#Train vs test accuracy plot 
train_test_accuracy_plot(boruta_fitted,X_train,y_train,model_name='Boruta')


#---------FEATURE SELECTION WITH DECISION TREE-----------------

# Perform feature selection using a decision tree model and get the sorted names of the selected features.
sorted_names = feat_selection_decision_tree(data, X_train, y_train, best_selected)

# Initialize lists to store accuracy and standard deviation for each feature subset.
acc_per_feature = []
std_per_feature = []

# Loop through the sorted feature names along with their index.
for index, feature_name in enumerate(sorted_names, start=1):

    # Get the subset of features up to the current index.
    input_features_names = sorted_names[:index]
    # Get the corresponding column indices of the selected features.
    input_features_index = [data.columns.get_loc(c) for c in input_features_names if c in data]

    # Fit the best selected model using the current subset of features.
    feature_model = best_selected.fit(X_train[:, input_features_index], y_train)

    # Save the model in the 'modelos' dictionary with a key based on the number of features used.
    modelos['number_features_' + str(index)] = feature_model

    # Subset the training data to include only the current features.
    X = X_train[:, input_features_index]

    # Calculate the classification report scores on the training and testing data using the feature_model.
    # 'classif_report' is a custom function for calculating classification report metrics.
    scores = classif_report(X, y_train, X_test, y_test, feature_model, print_report=False)

    # Calculate the mean accuracy and standard deviation of the classification scores for the current feature subset.
    acc_per_feature.append(np.mean(scores))
    std_per_feature.append(np.std(scores))


#Plot the learning curve
plot_learning_curve(acc_per_feature,std_per_feature,save=True,path_plot=path_plot+'/'+'features_plot_all.png')

#arranges the accuracy by feature in ascending order and takes the last value [-1] index
pos_model = np.argsort(acc_per_feature[1:])[-1] 
#search for the key corresponding to the best model index and saves it in best_model
best_model = list(modelos.keys())[pos_model]

#Save the model
#path_job = path_eli
path_job = path_save
joblib.dump(modelos[best_model], path_job+'/'+'modelo_entrenado_DT.pkl')

#saves best features in a .txt file
best_features=sorted_names[1:pos_model]
mi_path = path_plot+'/'+'best_features.txt'
f = open(mi_path, 'w')

for i in best_features:
    f.write(i+'\n')
f.close()

#create new dataframe with the best features only
new_data = pd.DataFrame()
for i in range(0,len(data.columns)):
    for j in range(0,len(best_features)):
        if data.columns[i] == best_features[j]:
            new_data[best_features[j]] = data[best_features[j]]



new_name = 'Data_complete_best_'+neuro+'_'+space+'_'+name+'_'+var
new_data.reset_index(drop=True).to_feather(path_save + rf'\dataframes\ML\{new_name}.feather')

#Show the correlation heatmap of the best features
heat_map(new_data,save=True,path_plot=path_plot+'/'+'correlation_randomforest.png')

#Learning curve

acc = []
std = []

for index, feature_name in enumerate(best_features,start=1):

    input_features_best = best_features[:index]
    input_best_index = [data.columns.get_loc(c) for c 
                      in input_features_best if c in data]
    fbest_model = best_selected.fit(X_train[:, input_best_index], y_train)

    scores_best = classif_report(X, y_train, X_test, y_test, feature_model, print_report=False)

    acc.append(np.mean(scores_best))
    std.append(np.std(scores_best))


#Plot learning curve
plot_learning_curve(acc,std)


predicted = fbest_model.predict(X_test[:,input_best_index])
classes_x=(predicted >= 0.5).astype(int)
#Compute precision and recall
computerprecision(y_test,classes_x)
#Confusion matrix
class_names=['G1','Control']
cm_test = confusion_matrix(y_test,classes_x)
confusion_matrix(path_plot,cm_test,classes=class_names,title='Confusion matrix')



#SVM (Grid search)
svm_GS_fitted, best_model = svm_grid_search(X_train, y_train)

acc_per_feature_svm = []
std_per_feature_svm = []

for index, feature_name in enumerate(sorted_names,start=1):

    input_features_best = best_features[:index]
    input_best_index = [data.columns.get_loc(c) for c 
                      in input_features_best if c in data]
    fbest_model = best_selected.fit(X_train[:, input_best_index], y_train)

    scores_best = classif_report(X, y_train, X_test, y_test, feature_model, print_report=False)

    acc.append(np.mean(scores_best))
    std.append(np.std(scores_best))

#The functions created in the _funciones file are imported
#from ML_models_functions import computerprecision,confusion_matrix,depuration,heat_map, classif_report, random_forest,train_test_accuracy_plot, boruta, feat_selection_decision_tree, plot_learning_curve, svm_grid_search, tpot
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
tk.Tk().withdraw() # part of the import if you are not using other tkinter functions

path_save = askdirectory()
print("user chose", path_save, "for save")
#path_save = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados_Eli'
neuro = 'sovaHarmony' #database
name = 'G1' #group
space = 'ic' #space
var = ''

path_save = os.path.join(path_save,'Resultados')
path_plot = path_save +rf'\graphics\ML/{neuro}/{name}_{var}_{space}'

path = askdirectory()
print("user chose", path_save, "for read feather")
path_df = rf'{path}\Data_complete_{space}_{neuro}_{name}.feather'

data = pd.read_feather(path_df)

#Dictionary with the models best predicted values 
modelos = {}

#-------DATA DEPURATION---------------------------------------------------------------


m = ['power','sl','cohfreq','entropy','crossfreq'] #metrics / features
b=['Gamma']
bm = ['Mdelta','Mtheta','Malpha-1','Malpha-2','Mbeta1','Mbeta2','Mbeta3','Mgamma'] #modulated crossed frequency bands

data = depuration(data,space,m) #new dataset


#----------TRAIN TEST SPLIT-------------------------------
X = data.values[:,:-1] #all the rows except the last one
y = data.values[:,-1] #all the columns except the last one

X_train, X_test, y_train, y_test = train_test_split(                            
X, # Valores de X
y, # Valores de Y
test_size=0.2, # Test de 20%
random_state=1, # Semilla
stratify=data.values[:,-1]) # que se mantenga la proporcion en la división

#-------HEAT MAP------------------
#heat_map(data)

#------------RANDOM FOREST -----------------------------------------

#Grid search fitted predict
GS_fitted,best_selected = random_forest(X_train, y_train)
'''print(GS_fitted)
 
#Add to the models dictionary
modelos['GridSerach'] = GS_fitted

#Print the classification report using the function
classif_report(X_train, y_train,X_test,y_test,GS_fitted)'''

#Train vs test accuracy plot
train_test_accuracy_plot(GS_fitted,X_train,y_train,model_name='Grid search')


#--------------BORUTA------------------------------------
boruta_fitted, X_transform = boruta(data,X_train,y_train,best_selected)

#Add to the models dictionary
modelos['Boruta'] = boruta_fitted

#Classification report
classif_report(X_transform, y_train,X_test,y_test,boruta_fitted)

#Train vs test accuracy plot 
train_test_accuracy_plot(boruta_fitted,X_train,y_train,model_name='Boruta')


#---------FEATURE SELECTION WITH DECISION TREE-----------------

# Perform feature selection using a decision tree model and get the sorted names of the selected features.
sorted_names = feat_selection_decision_tree(data, X_train, y_train, best_selected)

# Initialize lists to store accuracy and standard deviation for each feature subset.
acc_per_feature = []
std_per_feature = []

# Loop through the sorted feature names along with their index.
for index, feature_name in enumerate(sorted_names, start=1):

    # Get the subset of features up to the current index.
    input_features_names = sorted_names[:index]
    # Get the corresponding column indices of the selected features.
    input_features_index = [data.columns.get_loc(c) for c in input_features_names if c in data]

    # Fit the best selected model using the current subset of features.
    feature_model = best_selected.fit(X_train[:, input_features_index], y_train)

    # Save the model in the 'modelos' dictionary with a key based on the number of features used.
    modelos['number_features_' + str(index)] = feature_model

    # Subset the training data to include only the current features.
    X = X_train[:, input_features_index]

    # Calculate the classification report scores on the training and testing data using the feature_model.
    # 'classif_report' is a custom function for calculating classification report metrics.
    scores = classif_report(X, y_train, X_test, y_test, feature_model, print_report=False)

    # Calculate the mean accuracy and standard deviation of the classification scores for the current feature subset.
    acc_per_feature.append(np.mean(scores))
    std_per_feature.append(np.std(scores))


#Plot the learning curve
plot_learning_curve(acc_per_feature,std_per_feature,save=True,path_plot=path_plot+'/'+'features_plot_all.png')

#arranges the accuracy by feature in ascending order and takes the last value [-1] index
pos_model = np.argsort(acc_per_feature[1:])[-1] 
#search for the key corresponding to the best model index and saves it in best_model
best_model = list(modelos.keys())[pos_model]

#Save the model
#path_job = path_eli
path_job = path_save
joblib.dump(modelos[best_model], path_job+'/'+'modelo_entrenado_DT.pkl')

#saves best features in a .txt file
best_features=sorted_names[1:pos_model]
mi_path = path_plot+'/'+'best_features.txt'
f = open(mi_path, 'w')

for i in best_features:
    f.write(i+'\n')
f.close()

#create new dataframe with the best features only
new_data = pd.DataFrame()
for i in range(0,len(data.columns)):
    for j in range(0,len(best_features)):
        if data.columns[i] == best_features[j]:
            new_data[best_features[j]] = data[best_features[j]]



new_name = 'Data_complete_best_'+neuro+'_'+space+'_'+name+'_'+var
new_data.reset_index(drop=True).to_feather(path_save + rf'\dataframes\ML\{new_name}.feather')

#Show the correlation heatmap of the best features
heat_map(new_data,save=True,path_plot=path_plot+'/'+'correlation_randomforest.png')

#Learning curve

acc = []
std = []

for index, feature_name in enumerate(best_features,start=1):

    input_features_best = best_features[:index]
    input_best_index = [data.columns.get_loc(c) for c 
                      in input_features_best if c in data]
    fbest_model = best_selected.fit(X_train[:, input_best_index], y_train)

    scores_best = classif_report(X, y_train, X_test, y_test, feature_model, print_report=False)

    acc.append(np.mean(scores_best))
    std.append(np.std(scores_best))


#Plot learning curve
plot_learning_curve(acc,std)


predicted = fbest_model.predict(X_test[:,input_best_index])
classes_x=(predicted >= 0.5).astype(int)
#Compute precision and recall
computerprecision(y_test,classes_x)
#Confusion matrix
class_names=['G1','Control']
cm_test = confusion_matrix(y_test,classes_x)
confusion_matrix(path_plot,cm_test,classes=class_names,title='Confusion matrix')



#SVM (Grid search)
svm_GS_fitted, best_model = svm_grid_search(X_train, y_train)

acc_per_feature_svm = []
std_per_feature_svm = []

for index, feature_name in enumerate(sorted_names,start=1):

    input_features_best = best_features[:index]
    input_best_index = [data.columns.get_loc(c) for c 
                      in input_features_best if c in data]
    fbest_model = best_selected.fit(X_train[:, input_best_index], y_train)

    scores_best = classif_report(X, y_train, X_test, y_test, feature_model, print_report=False)

    acc.append(np.mean(scores_best))
    std.append(np.std(scores_best))