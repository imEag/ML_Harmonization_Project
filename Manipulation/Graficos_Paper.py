"""
This code is intended for use in a debugging setting, enabling independent generation of plots. 
Leveraging libraries like Pandas, Matplotlib, and Seaborn, it facilitates the manipulation and
visualization of data pertaining to brain signals, with a focus on the Beta3 band.

Data Loading and Preprocessing
Loads data from a feather file using Pandas
Replaces specific values in columns with new values
Calculates the number of rows and columns needed for plotting
Boxplots
Creates boxplots for each component (C1-C9) using Seaborn
Customizes plot appearance, including title, labels, and legend
Subject Table
Replaces specific values in columns with new values
Groups data by 'Database' and 'Group' and calculates statistics (count, mean age, standard deviation, and sex distribution)
Formats age column to display mean and standard deviation
Selects and reorganizes columns for the final table
Calculates total participants and sex distribution
Creates a total row and adds it to the table
Saves the table to an Excel file
Confusion Matrix
Calculates precision, recall, F1-score, and accuracy from confusion matrix values
Plots the confusion matrix using Matplotlib
Customizes plot appearance, including title, labels, and normalization
Plotting Functions
Defines a function to plot the confusion matrix with customization options
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[]
# Debbug data
str_ratio = '2to1'
file_path = os.path.join(path, neuro, f'integration{str_ratio}', space, name, f'Data_integration_ic_{neuro}_{name}.feather')
data = pd.read_feather(file_path)
data2to1 = data

# In[] CAJAS Y BIGOTES
# Definir los componentes C
components = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
# Reemplazar los valores específicos en las columnas correspondientes
data2to1['database'] = data2to1['database'].replace({'BIOMARCADORES': 'UdeA1', 'DUQUE': 'UdeA2'})
data2to1['group'] = data2to1['group'].replace({'G1': 'ACr'})
data2to1['group'] = data2to1['group'].replace({'Control': 'HC'})

# Calcular el número de filas y columnas necesarias para organizar los gráficos
n_rows = 3
n_cols = 3

# Crear un gráfico de cajas y bigotes para cada componente C
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 15))

# Definir paleta de colores
palette = ["#8AA6A3", "#127369", "#10403B", "#45C4B0"]

# Inicializar una lista para manejar las leyendas
handles, labels = [], []

# Iterar sobre los componentes C y generar los gráficos correspondientes
for i, comp in enumerate(components):
    row = i // n_cols
    col = i % n_cols
    df_comp = data2to1[[f'power_{comp}_Beta3', 'group', 'database']]
    ax = sns.boxplot(data=df_comp, x='group', y=f'power_{comp}_Beta3', hue='database', ax=axes[row, col], palette=palette)
    ax.set_title(f'Component = {comp}', fontsize=20)
    ax.set_xlabel('Group', fontsize=20)
    ax.set_ylabel('Relative Power', fontsize=20)
    ax.tick_params(axis='x', labelrotation=45, labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend().remove()  # Eliminar la leyenda individual de cada gráfico
    h, l = ax.get_legend_handles_labels()  # Obtener manijas y etiquetas para cada gráfico
    handles.extend(h)
    labels.extend(l)

# Añadir una única leyenda fuera de los gráficos
fig.legend(handles, labels[:4], loc='upper center', title='Cohorts', ncol=4, bbox_to_anchor=(0.5, 0.9), fontsize=20, title_fontsize=20)

# Eliminar gráficos vacíos si el número de componentes es menor que 9
if len(components) < n_rows * n_cols:
    for i in range(len(components), n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

# Ajustar el espacio superior para el título y guardar el gráfico
plt.suptitle('Comparison of the components (ICs) of the Beta3 band in four cohorts of interest', fontsize=20, y=0.92)
plt.tight_layout(rect=[0, 0.03, 1, 0.88])  # Ajustar el espacio superior para el título
plt.savefig(r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\PRODUCTOS\paper2_harmonization\grafico_bandas_beta3_neuroharmonize.png')


# In[] TABLA DE SUJETOS
# Reemplazar los valores específicos en las columnas correspondientes
data2to1['database'] = data2to1['database'].replace({'BIOMARCADORES': 'UdeA1', 'DUQUE': 'UdeA2'})
data2to1['group'] = data2to1['group'].replace({'G1': 'ACr'})

# Agrupar los datos por 'Database' y 'Group' y calcular las estadísticas
grouped = data2to1.groupby(['database', 'group']).agg(
    count=('age', 'size'),
    mean_age=('age', 'mean'),
    std_age=('age', 'std'),
    F=('sex', lambda x: (x == 'F').sum()),
    M=('sex', lambda x: (x == 'M').sum())
).reset_index()

# Formatear la columna de edad (Mean ± SD)
grouped['age'] = grouped['mean_age'].round(2).astype(str) + " ± " + grouped['std_age'].round(2).astype(str)
grouped['sex'] = grouped['F'].astype(str) + "/" + grouped['M'].astype(str)

# Seleccionar y reorganizar las columnas necesarias
grouped = grouped[['database', 'group', 'count', 'age', 'sex']]

# Calcular el total de participantes y la distribución por sexo
total_count = data2to1['sex'].count()
total_f = (data2to1['sex'] == 'F').sum()
total_m = (data2to1['sex'] == 'M').sum()

# Crear una fila de totales
total_row = pd.DataFrame([['Total', '', total_count, '', f'{total_f}/{total_m}']], columns=['database', 'group', 'count', 'age', 'sex'])

# Añadir la fila de totales a la tabla
final_table = pd.concat([grouped, total_row], ignore_index=True)

# Cambiar títulos para que empiecen con mayúscula
final_table.columns = ['Database', 'Group', 'Count', 'Age', 'Sex']

# Guardar la tabla final en un archivo de Excel
final_table.to_excel(r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\PRODUCTOS\paper2_harmonization\resumen_participantes_2to1.xlsx', index=False)

# Mostrar la tabla final
print(final_table)

# In[]
# Tomar los valores de la matriz de confusión
TP = 32
TN = 1
FP = 0
FN = 2

# Calcular precision y recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("Accuracy:", accuracy)
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", F1)

# In[]
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, ratio, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.BuGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Confusion Matrix normalize {ratio}")
    else:
        print(f'Confusion Matrix {ratio}')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     fontsize=20,  # Tamaño de fuente ajustado a 20
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

# Example usage
# Confusion matrix values (replace these values with yours)
cm = np.array([[31, 1],
               [1, 15]])
#cm = np.array([[31, 1],
#               [2, 4]])
#cm = np.array([[32, 0],
#               [2, 1]])

# Classes (replace these classes with yours)
classes = ['HC', 'ACr']

# Call the function to plot the confusion matrix
plt.figure(figsize=(10, 8))
ratio = '2:1'
plot_confusion_matrix(cm, ratio, classes, normalize=False, title=f'Confusion Matrix {ratio}', cmap=plt.cm.BuGn)
plt.savefig(r'C:\Users\veroh\OneDrive - Universidad de Antioquia\Articulo análisis longitudinal\PRODUCTOS\paper2_harmonization\CM2to1.png', bbox_inches='tight')
