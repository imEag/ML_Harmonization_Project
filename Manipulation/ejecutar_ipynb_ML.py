"""
Modulo de entrenamiento para diferentes pipelines de ML para el
entrenamiento utilizando los datasets creados. El flujo es capaz de 
generalizar en la mayoría de los casos y únicamente es necesario
realizar modificaciones en la ruta a los datos.

"""
import os
import pandas as pd 
import seaborn as sns                                                   
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


path_santiago = r'C:\Users\santi\Universidad de Antioquia\VALERIA CADAVID CASTRO - Resultados_Armonizacion_BD'
path_veronica = r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados\dataframes'
path_save =r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Verónica Henao Isaza\Resultados'
neuro = 'sovaHarmony'
name = 'G1'
space = 'ic'
var = ''
path_plot = path_save +rf'\graphics\ML/{neuro}/{name}_{var}_{space}'
os.makedirs(path_plot,exist_ok=True)

path = path_veronica
path_df = rf'{path}\{neuro}\integration\{space}\{name}\Data_complete_{space}_{neuro}_{name}.feather'

path_excel = path_save + rf'\tables\ML\{space}\{name}'
os.makedirs(path_excel,exist_ok=True)

path_excel1 = path_excel + rf'\describe_all_{var}.xlsx'
path_excel2 = path_excel + rf'\describe_{var}.xlsx'